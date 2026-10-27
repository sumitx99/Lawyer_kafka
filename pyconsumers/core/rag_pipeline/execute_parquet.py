"""Convert uploaded CSV to string-typed Parquet and a bounded preview."""
import asyncio
import csv
from hashlib import sha256
from io import StringIO

import pyarrow as pa
import pyarrow.csv as arrow_csv
import pyarrow.parquet as parquet

from .execute_standardization import decode_text
from .userallocation.UserAllocation import StageError, StageResult


class CsvParquet:
    def __init__(self, storage, settings):
        self.storage, self.settings = storage, settings

    async def __call__(self, claim):
        row = claim.row
        raw = await self.storage.get_bytes(row['s3_path'])
        if sha256(raw).hexdigest() != row['sha256']:
            raise StageError('Stored file does not match its upload checksum.', terminal=True)
        converted = await asyncio.to_thread(self.convert, raw)
        if converted is None:
            return StageResult(status='INAPPLICABLE')
        data, summary = converted
        key = row['s3_path'].rsplit('/', 1)[0] + '/parquet/data.parquet'
        await self.storage.put_bytes(key, data, 'application/vnd.apache.parquet')
        return StageResult(status='COMPLETED', summary={'parquet_key': key, **summary})

    def convert(self, raw):
        text, encoding = decode_text(raw)
        if not text.strip():
            return None
        try:
            delimiter = csv.Sniffer().sniff(text[:65536], delimiters=',;\t|').delimiter
        except csv.Error:
            delimiter = ','
        try:
            records = csv.reader(StringIO(text), delimiter=delimiter, strict=True)
            names = next(record for record in records if record)
            if len(names) > self.settings.csv_max_columns:
                raise StageError('CSV exceeds the column limit.', terminal=True)
            if len(set(names)) != len(names):
                raise StageError('CSV column names must be unique.', terminal=True)
            if next((record for record in records if record), None) is None:
                return None
            reader = arrow_csv.open_csv(
                pa.BufferReader(text.encode('utf-8')),
                parse_options=arrow_csv.ParseOptions(delimiter=delimiter, newlines_in_values=True),
                convert_options=arrow_csv.ConvertOptions(column_types={name: pa.string() for name in names},
                                                         strings_can_be_null=True, null_values=['']))
            batches, count = [], 0
            for batch in reader:
                count += batch.num_rows
                if count > self.settings.csv_max_rows:
                    raise StageError('CSV exceeds the row limit.', terminal=True)
                batches.append(batch)
            if not count:
                return None
            table = pa.Table.from_batches(batches, schema=reader.schema)
        except (pa.ArrowInvalid, csv.Error, StopIteration):
            raise StageError('CSV is malformed or cannot be read.', terminal=True) from None
        sink = pa.BufferOutputStream()
        parquet.write_table(table, sink, compression='zstd')
        # Inference is descriptive only; the stored schema always remains string.
        inferred = {}
        for name in names:
            values = [v for v in table[name].slice(0, 1000).to_pylist() if v is not None]
            kind = 'null'
            if values:
                kind = 'string'
                for candidate, parse in [('int64', int), ('float64', float)]:
                    try:
                        for value in values:
                            parse(value)
                    except ValueError:
                        continue
                    kind = candidate
                    break
            inferred[name] = kind
        sample = [{name: value[:500] if value is not None else None for name, value in row.items()}
                  for row in table.slice(0, 5).to_pylist()]
        return sink.getvalue().to_pybytes(), {
            'rows': count, 'columns': len(names), 'column_names': names,
            'null_counts': {name: table[name].null_count for name in names},
            'encoding': encoding, 'delimiter': delimiter, 'sample_rows': sample,
            'inferred_types': inferred, 'inference_sample_rows': min(count, 1000)}
