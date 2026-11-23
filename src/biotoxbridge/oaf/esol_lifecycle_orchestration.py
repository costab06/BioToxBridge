"""OAF completion-driven ESOL lifecycle. No modifications to OAF itself."""
from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from biotoxbridge.oaf.artifacts import publish_json, read_json
from biotoxbridge.oaf.agent import EVENT_COMPLETED, EVENT_REQUESTED
from biotoxbridge.oaf.esol_workflows import lifecycle_stages
from biotoxbridge.oaf.lifecycle import Lifecycle, LifecycleStep

STREAM = 'biotoxbridge.esol.events'
STAGES = ('esol_acquire', 'esol_train', 'esol_promote')


@contextmanager
def promotion_lock(store):
    """Serialize promotions between workers on one shared filesystem/host."""
    path = Path(store) / 'models' / '.promotion.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def safe_lifecycle_stages(store, *, tox21_url='http://localhost:8080'):
    stages = lifecycle_stages(store, tox21_url=tox21_url)
    original = stages['esol_promote']

    def locked_promote(request):
        with promotion_lock(store):
            return original(request)

    stages['esol_promote'] = locked_promote
    return stages

ESOL_LIFECYCLE = Lifecycle((
    LifecycleStep(
        "esol_acquire",
        "esol_train",
        lambda output, config: {
            "dataset_sha256": output["sha256"],
            "seed": config["seed"],
        },
    ),
    LifecycleStep(
        "esol_train",
        "esol_promote",
        lambda output, config: {
            "candidate_version": output["candidate_version"],
            **config["policy"],
        },
    ),
    LifecycleStep("esol_promote", None),
))


def next_request(stage, output, config):
    return ESOL_LIFECYCLE.next_request(stage, output, config)

def create_coordinator_class():
    from shared.agent.base import BaseAgent
    from shared.eventbus.events import EventEnvelope

    class ESOLLifecycleCoordinator(BaseAgent):
        def __init__(self, *, event_bus, correlation_registry, artifact_root, name='biotoxbridge-esol-lifecycle', **kwargs):
            super().__init__(name=name, event_bus=event_bus,
                             correlation_registry=correlation_registry,
                             subscribed_event_types={EVENT_COMPLETED}, **kwargs)
            self.artifact_root = Path(artifact_root)

        async def handle_event(self, event):
            payload = event.payload
            if not isinstance(payload, dict):
                raise ValueError('Completion payload must be an object')
            run_id, stage = payload['run_id'], payload['stage']
            if stage not in STAGES:
                return []  # inference completion is unrelated to training lifecycle
            config = read_config(self.artifact_root, run_id)
            if not config.get('lifecycle_v1'):
                return []
            output = read_json(payload['output_artifact'], self.artifact_root)
            next_stage, request = next_request(stage, output, config)
            if next_stage is None:
                # Completion event itself is durable; accepted pointer is managed by registry.
                return []
            request_id = f'lifecycle-{next_stage}-input'
            input_ref = publish_json(self.artifact_root, run_id, request_id, request)
            event_id = uuid5(NAMESPACE_URL, f'biotoxbridge:lifecycle:{run_id}:{next_stage}')
            return [EventEnvelope(event_id=event_id, event_type=EVENT_REQUESTED,
                                  source_agent=str(self.name), correlation_id=event.correlation_id,
                                  causation_id=event.event_id,
                                  payload={'run_id': run_id, 'stage': next_stage,
                                           'input_artifact': input_ref})]

    return ESOLLifecycleCoordinator


def read_config(root, run_id):
    # The expected reference is reconstructed from the immutable artifact's bytes.
    # read_json verifies the checksum and path after reconstruction.
    from hashlib import sha256
    from biotoxbridge.oaf.artifacts import _safe_path
    path = _safe_path(Path(root).resolve(), run_id, 'lifecycle-config')
    raw = path.read_bytes()
    ref = {'artifact_id': f'{run_id}/lifecycle-config.json', 'uri': str(path),
           'checksum': sha256(raw).hexdigest()}
    return read_json(ref, root)
