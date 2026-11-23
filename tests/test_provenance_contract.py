import hashlib
import json
import pytest
from biotoxbridge.provenance import write_run_manifest
from biotoxbridge.provenance import verify_run_manifest


def test_manifest_hashes_inputs_outputs_and_metadata(tmp_path):
    config = tmp_path / 'config.yaml'
    source = tmp_path / 'source.csv'
    split = tmp_path / 'split.json'
    config.write_text('experiment: test\n')
    source.write_bytes(b'chemical,score\na,0.5\n')
    split.write_text('{"train":[],"test":[]}')
    run = tmp_path / 'run'
    write_run_manifest(run, config, {}, {'split_manifest': split},
                       input_paths={'source': source},
                       model_metadata={'model_set_id': 'test-v1'})
    m = json.loads((run / 'run_manifest.json').read_text())
    assert m['inputs']['source']['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert m['artifacts']['split_manifest']['sha256'] == hashlib.sha256(split.read_bytes()).hexdigest()
    assert m['model_metadata']['model_set_id'] == 'test-v1'


@pytest.mark.parametrize('kind', ['input', 'output'])
def test_manifest_rejects_missing_required_file(tmp_path, kind):
    config = tmp_path / 'config.yaml'
    config.write_text('x: 1')
    missing = tmp_path / 'missing.csv'
    with pytest.raises(FileNotFoundError):
        write_run_manifest(tmp_path / 'run', config, {},
                           {'output': missing} if kind == 'output' else {},
                           input_paths={'input': missing} if kind == 'input' else {})
    assert not (tmp_path / 'run' / 'run_manifest.json').exists()

def test_manifest_verification_and_replay(tmp_path):
    config = tmp_path / "config.yaml"
    source = tmp_path / "source.csv"
    output = tmp_path / "output.csv"

    config.write_text("experiment: test\n")
    source.write_text("input\n")
    output.write_text("result\n")

    run = tmp_path / "run"

    kwargs = {
        "input_paths": {"source": source},
        "model_metadata": {"model_set_id": "v1"},
    }

    write_run_manifest(run, config, {}, {"output": output}, **kwargs)
    first = (run / "run_manifest.json").read_bytes()

    verified = verify_run_manifest(run)
    assert verified["schema_version"] == 1

    write_run_manifest(run, config, {}, {"output": output}, **kwargs)
    assert (run / "run_manifest.json").read_bytes() == first


def test_manifest_detects_modified_input(tmp_path):
    config = tmp_path / "config.yaml"
    source = tmp_path / "source.csv"

    config.write_text("experiment: test\n")
    source.write_text("original\n")

    run = tmp_path / "run"

    write_run_manifest(
        run, config, {}, {}, input_paths={"source": source}
    )

    source.write_text("modified\n")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_run_manifest(run)


def test_manifest_detects_modified_output(tmp_path):
    config = tmp_path / "config.yaml"
    output = tmp_path / "output.csv"

    config.write_text("experiment: test\n")
    output.write_text("original\n")

    run = tmp_path / "run"

    write_run_manifest(run, config, {}, {"output": output})
    output.write_text("modified\n")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_run_manifest(run)


def test_manifest_rejects_reserved_metadata(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("experiment: test\n")

    with pytest.raises(ValueError, match="reserved fields"):
        write_run_manifest(
            tmp_path / "run",
            config,
            {},
            {},
            extra={"config_sha256": "invalid"},
        )