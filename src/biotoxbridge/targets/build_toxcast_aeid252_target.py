#!/usr/bin/env python3
"""Freeze a chemical-level binary target for ToxCast AEID 252.

Uses the existing endpoint-specific extraction, never the multi-GB raw archive.
No external dependencies. All outputs are deterministic for identical input bytes.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

AEID = 252
FLAG_NAMES = {
    5: 'modl.directionality.fail', 6: 'singlept.hit.high',
    7: 'singlept.hit.mid', 8: 'multipoint.neg', 9: 'bmd.high',
    10: 'noise', 11: 'border', 13: 'low.nrep', 14: 'low.nconc',
    15: 'gnls.lowconc', 17: 'efficacy.50', 18: 'ac50.lowconc',
}
DTXSID = re.compile(r'^DTXSID\d+$', re.IGNORECASE)
FIELDS = ['dsstox_substance_id', 'aeid', 'hitcall', 'hitc', 'mc6_flags', 'spid', 'chid', 'casn', 'chnm']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def parse_label(value):
    s = str(value or '').strip()
    if s in ('0', '0.0'): return 0
    if s in ('1', '1.0'): return 1
    return None


def parse_flags(raw):
    raw = str(raw or '').strip()
    if not raw or raw.lower() in ('nan', 'none', 'null', '[]'):
        return set(), []
    tokens = re.findall(r'\d+', raw)
    flags = {int(t) for t in tokens}
    return flags, sorted(flags - FLAG_NAMES.keys())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('data/interim/toxcast_aeid252_extraction_v1/toxcast_activity_records.csv'))
    p.add_argument('--output-dir', type=Path, default=Path('data/interim/toxcast_aeid252_target_v1'))
    args = p.parse_args()
    src = args.input
    if not src.is_file():
        p.error(f'Input file not found: {src}')
    if args.output_dir.resolve() == src.parent.resolve():
        p.error('Output directory must differ from input directory')

    grouped = defaultdict(list)
    excluded = []
    counts = Counter()
    hitc_mismatches = []
    observed_flags = Counter()
    unknown_flags = Counter()

    with src.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        missing = {'dsstox_substance_id', 'hitcall', 'aeid', 'mc6_flags'} - set(reader.fieldnames or [])
        if missing:
            p.error(f'Missing required input columns: {sorted(missing)}')
        for line_number, row in enumerate(reader, start=2):
            counts['source_rows'] += 1
            try:
                row_aeid = int(str(row['aeid']).strip())
            except (ValueError, TypeError):
                raise ValueError(f'Invalid AEID on CSV line {line_number}')
            if row_aeid != AEID:
                raise ValueError(f'Unexpected AEID {row_aeid} on CSV line {line_number}; expected {AEID}')
            dtxsid = str(row.get('dsstox_substance_id') or '').strip().upper()
            label = parse_label(row.get('hitcall'))
            flags, unknown = parse_flags(row.get('mc6_flags'))
            for flag in flags: observed_flags[flag] += 1
            for flag in unknown: unknown_flags[flag] += 1
            record = {'line': line_number, 'dtxsid': dtxsid, 'label': label, 'flags': flags,
                      'hitc': str(row.get('hitc') or '').strip(), 'row': row}
            if not DTXSID.fullmatch(dtxsid):
                counts['invalid_identifier_rows'] += 1
                excluded.append({'dsstox_substance_id': dtxsid, 'reason': 'invalid_dtxsid',
                                 'record_count': 1, 'hitcalls': str(row.get('hitcall') or ''),
                                 'source_lines': str(line_number), 'mc6_flags': ','.join(map(str, sorted(flags)))})
                continue
            if label is None:
                counts['invalid_label_rows'] += 1
            if record['hitc']:
                try:
                    continuous = float(record['hitc'])
                    if not 0 <= continuous <= 1 or (label is not None and int(continuous >= 0.9) != label):
                        hitc_mismatches.append({'line': line_number, 'dtxsid': dtxsid,
                                                'hitc': record['hitc'], 'hitcall': row['hitcall']})
                except ValueError:
                    hitc_mismatches.append({'line': line_number, 'dtxsid': dtxsid,
                                            'hitc': record['hitc'], 'hitcall': row['hitcall']})
            grouped[dtxsid].append(record)

    targets = []
    detail = []
    for dtxsid in sorted(grouped):
        records = grouped[dtxsid]
        labels = {r['label'] for r in records}
        flags = sorted(set().union(*(r['flags'] for r in records)))
        lines = ','.join(str(r['line']) for r in records)
        labels_text = ','.join(str(x) for x in sorted(labels, key=lambda v: -1 if v is None else v))
        if None in labels:
            reason = 'invalid_hitcall'
        elif len(labels) != 1:
            reason = 'conflicting_hitcalls'
        else:
            reason = None
        if reason:
            counts[reason + '_chemicals'] += 1
            excluded.append({'dsstox_substance_id': dtxsid, 'reason': reason,
                             'record_count': len(records), 'hitcalls': labels_text,
                             'source_lines': lines, 'mc6_flags': ','.join(map(str, flags))})
            continue
        label = next(iter(labels))
        targets.append({'dsstox_substance_id': dtxsid, 'label': label,
                        'source_record_count': len(records), 'mc6_flags': ','.join(map(str, flags)),
                        'has_quality_flags': int(bool(flags)),
                        'has_low_replication': int(13 in flags),
                        'has_low_concentration_count': int(14 in flags),
                        'has_other_flags': int(bool(set(flags) - {13, 14}))})
        for r in records:
            detail.append({'dsstox_substance_id': dtxsid, 'label': label,
                           'source_csv_line': r['line'], 'hitc': r['hitc'],
                           'hitcall': r['row'].get('hitcall', ''),
                           'mc6_flags': ','.join(map(str, sorted(r['flags']))),
                           'spid': r['row'].get('spid', ''), 'chid': r['row'].get('chid', ''),
                           'casn': r['row'].get('casn', ''), 'chnm': r['row'].get('chnm', '')})

    # Fail closed if the continuous score contradicts the binary label.
    # Do not publish a frozen dataset whose activity semantics are unresolved.
    if hitc_mismatches:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / 'hitc_hitcall_discrepancies.csv'
        write_csv(path, ['line', 'dtxsid', 'hitc', 'hitcall'], hitc_mismatches)
        raise RuntimeError(f'{len(hitc_mismatches)} hitc/hitcall discrepancies; inspect {path} before freezing')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir
    write_csv(output / 'toxcast_aeid252_targets.csv',
              ['dsstox_substance_id', 'label', 'source_record_count', 'mc6_flags',
               'has_quality_flags', 'has_low_replication', 'has_low_concentration_count', 'has_other_flags'], targets)
    write_csv(output / 'toxcast_aeid252_exclusions.csv',
              ['dsstox_substance_id', 'reason', 'record_count', 'hitcalls', 'source_lines', 'mc6_flags'],
              sorted(excluded, key=lambda r: (r['dsstox_substance_id'], r['reason'])))
    write_csv(output / 'toxcast_aeid252_retained_source_records.csv',
              ['dsstox_substance_id', 'label', 'source_csv_line', 'hitc', 'hitcall', 'mc6_flags',
               'spid', 'chid', 'casn', 'chnm'], detail)
    flag_rows = [{'flag_id': flag, 'method': FLAG_NAMES.get(flag, 'UNKNOWN'), 'source_record_count': count}
                 for flag, count in sorted(observed_flags.items())]
    write_csv(output / 'toxcast_aeid252_flag_summary.csv',
              ['flag_id', 'method', 'source_record_count'], flag_rows)
    labels = Counter(r['label'] for r in targets)
    summary = {
        'aeid': AEID, 'endpoint': 'BSK_hDFCGF_Proliferation',
        'source_file': str(src), 'source_sha256': sha256(src),
        'source_rows': counts['source_rows'], 'unique_valid_identifiers': len(grouped),
        'retained_chemicals': len(targets), 'active_chemicals': labels[1],
        'inactive_chemicals': labels[0], 'active_prevalence': (labels[1] / len(targets) if targets else None),
        'excluded_conflicting_chemicals': counts['conflicting_hitcalls_chemicals'],
        'excluded_invalid_label_chemicals': counts['invalid_hitcall_chemicals'],
        'invalid_identifier_rows': counts['invalid_identifier_rows'],
        'unknown_flag_ids': dict(sorted(unknown_flags.items())),
        'label_policy': 'Use binary source hitcall (0/1), validate against hitc >= 0.9 where hitc supplied',
        'duplicate_policy': 'Collapse only records with identical binary labels per DTXSID; union flags',
        'quality_policy': 'Do not exclude on mc6 flags in primary dataset; preserve and stratify for sensitivity analyses',
        'structure_resolution_status': 'not yet performed',
        'output_files': {},
    }
    for filename in ('toxcast_aeid252_targets.csv', 'toxcast_aeid252_exclusions.csv',
                     'toxcast_aeid252_retained_source_records.csv', 'toxcast_aeid252_flag_summary.csv'):
        summary['output_files'][filename] = sha256(output / filename)
    (output / 'toxcast_aeid252_manifest.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f'\nOutput directory: {output.resolve()}')

if __name__ == '__main__':
    main()
