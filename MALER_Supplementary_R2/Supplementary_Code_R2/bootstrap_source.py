"""Restore the 15 compressed source assets and verify all 389 source hashes."""
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'


def checked_path(base, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe source asset path.')
    path = base.joinpath(*rel.parts)
    if base.resolve() not in path.resolve().parents:
        raise ValueError('Source asset path escapes its directory.')
    return path


def main():
    manifest = json.loads((ROOT / 'source_file_manifest.json').read_text(encoding='utf-8'))
    assets = json.loads((ROOT / 'source_assets.json').read_text(encoding='utf-8'))
    if assets['commit'] != manifest['commit']:
        raise ValueError('Asset and source commits differ.')
    expected = {r['file']: r for r in manifest['files']}
    source_path_map = assets.get('source_path_map', {})
    restored = 0
    for row in assets['assets']:
        target = checked_path(SOURCE, source_path_map.get(row['file'], row['file']))
        if target.exists():
            continue
        if expected[row['file']]['sha256'] != row['sha256']:
            raise ValueError('Asset is not in the locked source manifest.')
        packed = checked_path(ROOT, row['storage']).read_bytes()
        if hashlib.sha256(packed).hexdigest() != row['stored_sha256']:
            raise ValueError('Compressed asset checksum mismatch.')
        data = lzma.decompress(packed)
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Restored asset checksum mismatch.')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        restored += 1
    for row in manifest['files']:
        path = checked_path(SOURCE, source_path_map.get(row['file'], row['file']))
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Reviewed source checksum mismatch: ' + row['file'])
    result = {'recorded_source_commit': manifest['commit'], 'source_hashes_verified': len(expected),
              'restored_assets': restored, 'identity_method': '389-file SHA-256 manifest; no Git repository',
              'normalized_source_filenames': len(source_path_map), 'git_clean_status_claimed': False}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
