"""B1 of MFL051 (SOL-R2-B1), the part of `capacity.py`: a commitment handed over with `Lease.keep`.

The space commitment of an attempt is the durable condition of "the absence of its children is not attested
yet": its record in `operations/` exists from before the first child. `keep` lets the worker hold it past the
block that made it; these are the facts the release rule of the worker relies on (test_unattested_end.py).
"""
import os
import tempfile
import unittest

from markdown_for_llms import capacity
from markdown_for_llms.settings import Settings
from markdown_for_llms.storage import Storage, StorageError

TOKEN, OTHER = "a" * 32, "b" * 32
BUDGET = 6 * capacity.MIB
PATHS = ("attempts/" + TOKEN, "results/" + TOKEN)


class KeepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_root=self.tmp.name + "/data", model_root=self.tmp.name + "/models")
        self.storage = Storage(self.settings)
        self.addCleanup(self.storage.close)
        self.operations = self.storage.root / "operations"

    def records(self):
        return sorted(p.suffix for p in self.operations.iterdir())

    def committed(self):
        return sum(capacity.read_json(self.storage.root, p.relative_to(self.storage.root).as_posix())["budget"]
                   for p in self.operations.glob("*.json"))

    def owners(self):
        with self.storage.lock("admission.lock"):
            return sorted(capacity.live_workers(self.storage)), sorted(capacity.dead_workers(self.storage))

    def kept(self):
        with self.storage.reservation("worker", BUDGET, TOKEN, PATHS) as lease:
            kept = lease.keep()
        self.addCleanup(kept.abandon)
        return kept

    def test_without_keep_the_record_goes_when_the_block_ends_whatever_happened_in_it(self):
        """What the worker did before B1: the commitment was gone before the job got its terminal state."""
        with self.assertRaises(StorageError):
            with self.storage.reservation("worker", BUDGET, TOKEN, PATHS):
                self.assertEqual((self.records(), self.committed()), ([".json"], BUDGET))
                raise StorageError("Conversion failed: isolation_unavailable; source kept, retry available")
        self.assertEqual((self.records(), self.committed()), ([], 0))

    def test_a_kept_commitment_stays_and_its_budget_stays_committed(self):
        kept = self.kept()
        self.assertEqual((self.records(), self.committed()), ([".json"], BUDGET))
        self.assertEqual(self.owners(), ([TOKEN], []))
        # a request that fits only without the kept budget is refused
        with self.assertRaisesRegex(StorageError, "quota exceeded"):
            self.storage.capacity(self.settings.storage_mib * capacity.MIB - self.committed() + 1)
        # and no change of the configuration passes while it is there (storage.py: quiesce)
        with self.assertRaisesRegex(StorageError, "reconcile interrupted operations first"):
            self.storage.quiesce(self.storage.identity["config"])
        self.assertIsNotNone(kept.fd)

    def test_while_its_owner_lives_no_recover_collects_it(self):
        self.kept()
        with self.storage.lock("admission.lock"):
            capacity.recover(self.storage)                       # what every new reservation does first
            capacity.recover(self.storage, workers=True)         # what the reconciliation of a worker does
            capacity.recover(self.storage, workers={TOKEN})
        self.assertEqual(self.records(), [".json"])
        with self.storage.reservation("control", 65536):          # another operation goes on beside it
            self.assertEqual(self.records(), [".json", ".json"])
        self.assertEqual(self.records(), [".json"])

    def test_after_the_death_of_its_owner_only_the_worker_that_names_the_attempt_collects_it(self):
        kept = self.kept()
        kept.abandon()                                           # the worker died: its descriptor closed, the record stays
        self.assertIsNone(kept.fd)
        self.assertEqual((self.records(), self.committed()), ([".json"], BUDGET))
        self.assertEqual(self.owners(), ([], [TOKEN]))            # dead_workers is the complement of live_workers
        with self.storage.lock("admission.lock"):
            capacity.recover(self.storage)                       # any other operation: never without the worker
        self.assertEqual(self.records(), [".json"])
        with self.storage.lock("admission.lock"):
            capacity.recover(self.storage, workers={OTHER})      # the attestation of another attempt
        self.assertEqual(self.records(), [".json"])
        with self.storage.lock("admission.lock"):
            capacity.recover(self.storage, workers={TOKEN})      # after the attestation of this one
        self.assertEqual((self.records(), self.committed()), ([], 0))

    def test_the_owner_alive_releases_it_with_close(self):
        kept = self.kept()
        kept.close()
        self.assertEqual((self.records(), self.committed(), self.owners()), ([], 0, ([], [])))
        kept.close()                                             # twice is nothing

    def test_dead_workers_names_only_worker_commitments_without_an_owner(self):
        with self.storage.reservation("control", 65536):
            with self.storage.reservation("worker", BUDGET, OTHER, ("attempts/" + OTHER,)):
                self.assertEqual(self.owners(), ([OTHER], []))
        kept = self.kept()
        kept.abandon()
        fd = os.open(self.operations / next(self.operations.iterdir()).name, os.O_RDONLY)
        os.close(fd)
        self.assertEqual(self.owners(), ([], [TOKEN]))
        self.assertEqual(self.records(), [".json"])               # asking collects nothing


if __name__ == "__main__":
    unittest.main()
