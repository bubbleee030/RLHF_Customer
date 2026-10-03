#!/usr/bin/env python3
"""
Automated Argilla dataset backup + UTF-8 JSON encoding fixer.
"""

import argparse
import copy
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import argilla as rg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_BACKUP_DIR = PROJECT_DIR / "backups"
DEFAULT_ARGILLA_DIR = PROJECT_DIR / "argilla"


class ArgillaBackupManager:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        dataset_name: str,
        workspace_name: str,
        backup_dir: str,
        max_backups: int,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.dataset_name = dataset_name
        self.workspace_name = workspace_name
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self.client: Optional[rg.Argilla] = None
        self.dataset = None

    def connect(self) -> bool:
        try:
            logger.info("Connecting to Argilla: %s", self.api_url)
            self.client = rg.Argilla(api_url=self.api_url, api_key=self.api_key)
            self.dataset = self.client.datasets(name=self.dataset_name, workspace=self.workspace_name)
            if self.dataset is None:
                logger.error("Dataset not found: %s (workspace=%s)", self.dataset_name, self.workspace_name)
                return False
            logger.info("Connected to dataset: %s", self.dataset_name)
            return True
        except Exception as exc:
            logger.exception("Failed to connect: %s", exc)
            return False

    def create_backup_dir(self) -> None:
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def get_backup_path(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.backup_dir / f"{self.dataset_name}_{ts}"

    def get_existing_backups(self):
        if not self.backup_dir.exists():
            return []
        backups = sorted(
            [p for p in self.backup_dir.iterdir() if p.is_dir() and p.name != "latest"],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return backups

    @staticmethod
    def _rewrite_json_utf8(file_path: Path) -> bool:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            return True
        except Exception as exc:
            logger.warning("Failed to rewrite %s: %s", file_path, exc)
            return False

    def fix_backup_json_encoding(self, backup_path: Path) -> int:
        fixed = 0
        for json_file in backup_path.rglob("*.json"):
            if self._rewrite_json_utf8(json_file):
                fixed += 1
        logger.info("UTF-8 rewrite completed: %d file(s)", fixed)
        return fixed

    def fix_existing_json_encoding(self, target_dir: str) -> int:
        root = Path(target_dir)
        if not root.exists():
            logger.error("Path does not exist: %s", root)
            return 0
        fixed = 0
        for json_file in root.rglob("*.json"):
            if self._rewrite_json_utf8(json_file):
                fixed += 1
        logger.info("Fixed existing JSON files under %s: %d", root, fixed)
        return fixed

    @staticmethod
    def _extract_ranked_values(value_obj):
        if not isinstance(value_obj, list) or not value_obj:
            return None
        if not all(isinstance(item, dict) and "value" in item and "rank" in item for item in value_obj):
            return None

        ranked_items = [item for item in value_obj if item.get("rank") is not None]
        if not ranked_items:
            return [item.get("value") for item in value_obj]

        ranked_items.sort(key=lambda item: item["rank"])
        return [item.get("value") for item in ranked_items]

    def _build_ranking_correction_outputs(self, backup_path: Path, api_records) -> None:
        records_path = backup_path / "records.json"
        if not records_path.exists():
            logger.warning("records.json not found; skip ranking correction")
            return

        try:
            with records_path.open("r", encoding="utf-8") as f:
                sdk_records = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read records.json: %s", exc)
            return

        api_by_server_id = {rec.get("id"): rec for rec in api_records if rec.get("id")}
        corrected_records = copy.deepcopy(sdk_records)

        mismatch_report = {
            "total_records": len(corrected_records),
            "records_with_mismatch": 0,
            "total_mismatches": 0,
            "details": [],
        }

        for rec in corrected_records:
            server_id = rec.get("_server_id")
            if not server_id:
                continue

            api_rec = api_by_server_id.get(server_id)
            if not api_rec:
                continue

            sdk_responses = rec.get("responses")
            if not isinstance(sdk_responses, dict):
                continue

            mismatches_for_record = []
            responses_from_api = api_rec.get("responses", [])
            for response in responses_from_api:
                user_id = response.get("user_id")
                values = response.get("values", {})
                if not user_id or not isinstance(values, dict):
                    continue

                for question_name, question_payload in values.items():
                    if not isinstance(question_payload, dict):
                        continue

                    ranked_value = self._extract_ranked_values(question_payload.get("value"))
                    if ranked_value is None:
                        continue

                    sdk_answers = sdk_responses.get(question_name)
                    if not isinstance(sdk_answers, list):
                        continue

                    for answer in sdk_answers:
                        if answer.get("user_id") != user_id:
                            continue

                        sdk_value = answer.get("value")
                        if sdk_value != ranked_value:
                            mismatches_for_record.append(
                                {
                                    "question": question_name,
                                    "user_id": user_id,
                                    "sdk_value": sdk_value,
                                    "api_ranked_value": ranked_value,
                                }
                            )
                            answer["value"] = ranked_value
                        break

            if mismatches_for_record:
                mismatch_report["records_with_mismatch"] += 1
                mismatch_report["total_mismatches"] += len(mismatches_for_record)
                mismatch_report["details"].append(
                    {
                        "server_record_id": server_id,
                        "external_id": rec.get("id"),
                        "mismatches": mismatches_for_record,
                    }
                )

        corrected_path = backup_path / "records.fixed.json"
        with corrected_path.open("w", encoding="utf-8") as f:
            json.dump(corrected_records, f, ensure_ascii=False, indent=2)
            f.write("\n")

        report_path = backup_path / "ranking_mismatch_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(mismatch_report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        logger.info(
            "Ranking correction finished: %d mismatch(es) across %d record(s)",
            mismatch_report["total_mismatches"],
            mismatch_report["records_with_mismatch"],
        )

    @staticmethod
    def _calculate_records_hash(records_json: Path) -> Optional[str]:
        import hashlib

        if not records_json.exists():
            return None
        try:
            content = records_json.read_text(encoding="utf-8")
            return hashlib.sha256(content.encode("utf-8")).hexdigest()
        except Exception:
            return None

    def backup_dataset(self) -> bool:
        if self.dataset is None:
            logger.error("Dataset is not loaded")
            return False

        self.create_backup_dir()
        existing_backups = self.get_existing_backups()
        old_hash = None
        if existing_backups:
            old_hash = self._calculate_records_hash(existing_backups[0] / "records.json")

        backup_path = self.get_backup_path()
        logger.info("Creating backup: %s", backup_path)
        try:
            backup_path.mkdir(parents=True, exist_ok=False)
            self.dataset.to_disk(path=str(backup_path), with_records=True)
            self.fix_backup_json_encoding(backup_path)

            # --- PATCH: Fetch raw API records to bypass SDK ranking serialization bug ---
            try:
                import requests
                headers = {"X-Argilla-API-Key": self.api_key, "Content-Type": "application/json"}
                ds_res = requests.get(f"{self.api_url}/api/v1/me/datasets", headers=headers)
                if ds_res.status_code == 200:
                    ds_id = next(d["id"] for d in ds_res.json()["items"] if d["name"] == self.dataset_name)
                    
                    all_records = []
                    offset = 0
                    while True:
                        r = requests.get(f"{self.api_url}/api/v1/datasets/{ds_id}/records?offset={offset}&limit=100&include=responses&include=suggestions", headers=headers)
                        if r.status_code != 200: break
                        items = r.json().get("items", [])
                        if not items: break
                        all_records.extend(items)
                        offset += 100
                    import json
                    with open(backup_path / "records_api.json", "w", encoding="utf-8") as f:
                        json.dump(all_records, f, indent=2, ensure_ascii=False)
                        f.write("\n")

                    self._build_ranking_correction_outputs(backup_path, all_records)
                    logger.info("Successfully fetched %d raw API records to fix Ranking export format.", len(all_records))
            except Exception as e:
                logger.error("Failed to fetch raw API records: %s", e)
            # -----------------------------------------------------------------------------


            new_hash = self._calculate_records_hash(backup_path / "records.json")
            if old_hash is not None and old_hash == new_hash:
                logger.info("No content change detected; removing duplicate backup")
                shutil.rmtree(backup_path)
                self._ensure_latest_exists()
                return True

            metadata = {
                "timestamp": datetime.now().isoformat(),
                "dataset_name": self.dataset_name,
                "workspace": self.workspace_name,
                "records_hash": new_hash,
            }
            with (backup_path / "backup_metadata.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                f.write("\n")

            self._update_latest_copy()
            self._auto_commit_latest()
            logger.info("Backup completed")
            return True
        except FileExistsError:
            logger.error("Backup directory already exists: %s", backup_path)
            return False
        except Exception as exc:
            logger.exception("Backup failed: %s", exc)
            if backup_path.exists():
                shutil.rmtree(backup_path)
            return False

    def rotate_backups(self) -> bool:
        backups = self.get_existing_backups()
        if len(backups) <= self.max_backups:
            return True
        for backup in backups[self.max_backups :]:
            logger.info("Removing old backup: %s", backup)
            shutil.rmtree(backup)
        return True

    def _update_latest_copy(self) -> None:
        backups = self.get_existing_backups()
        if not backups:
            return
        latest_src = backups[0]
        latest_dst = self.backup_dir / "latest"
        if latest_dst.exists():
            shutil.rmtree(latest_dst)
        shutil.copytree(latest_src, latest_dst)
        logger.info("Updated latest backup copy")

    def _ensure_latest_exists(self) -> None:
        latest_dst = self.backup_dir / "latest"
        if latest_dst.exists():
            return
        backups = self.get_existing_backups()
        if not backups:
            return
        shutil.copytree(backups[0], latest_dst)

    def _auto_commit_latest(self) -> None:
        repo_root = self.backup_dir.parent
        try:
            check = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
            if check.returncode != 0:
                return

            status = subprocess.run(
                ["git", "status", "--porcelain", str(self.backup_dir / "latest")],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
            if not status.stdout.strip():
                return

            subprocess.run(["git", "add", str(self.backup_dir / "latest")], cwd=repo_root, check=True)
            msg = f"Backup updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            subprocess.run(["git", "commit", "-m", msg], cwd=repo_root, check=True)
            subprocess.run(["git", "push"], cwd=repo_root, check=True)
            logger.info("Auto-committed latest backup")
        except Exception as exc:
            logger.warning("Skipping git auto-commit: %s", exc)

    def run_once(self) -> bool:
        if not self.connect():
            return False
        if not self.backup_dataset():
            return False
        return self.rotate_backups()


def schedule_backups(manager: ArgillaBackupManager, interval_minutes: int) -> None:
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        logger.error("APScheduler not installed. Run: pip install apscheduler")
        return

    scheduler = BackgroundScheduler()
    scheduler.add_job(manager.run_once, "interval", minutes=interval_minutes, id="argilla_backup")
    scheduler.start()
    logger.info("Scheduler started: every %s minutes", interval_minutes)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping scheduler")
        scheduler.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automated Argilla backup + JSON UTF-8 fixer")
    parser.add_argument(
        "action",
        nargs="?",
        choices=["backup", "schedule", "list", "fix-encoding"],
        default="backup",
        help="Action to run (default: backup)",
    )
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL", "https://your-argilla-server.hf.space"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"), help="Argilla API key")
    parser.add_argument("--dataset", default=os.getenv("ARGILLA_DATASET", "TAIWAN_AI_RAP_Helpfulness"))
    parser.add_argument("--workspace", default=os.getenv("ARGILLA_WORKSPACE", "argilla"))
    parser.add_argument("--backup-dir", default=os.getenv("ARGILLA_BACKUP_DIR", str(DEFAULT_BACKUP_DIR)))
    parser.add_argument("--max-backups", type=int, default=int(os.getenv("ARGILLA_MAX_BACKUPS", "5")))
    parser.add_argument("--schedule", type=int, help="Run periodically in minutes")
    parser.add_argument("--once", action="store_true", help="Run one backup cycle")
    parser.add_argument("--list", action="store_true", help="List backups")
    parser.add_argument("--fix-existing", help="Fix all JSON files under this path and exit")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Allow both style: positional action and legacy flags.
    if args.action == "schedule" and args.schedule is None:
        args.schedule = 120
    if args.action == "list":
        args.list = True
    if args.action == "fix-encoding" and not args.fix_existing:
        args.fix_existing = str(DEFAULT_ARGILLA_DIR)
    if args.action == "backup":
        args.once = True

    # API key is required only for Argilla API operations.
    needs_api = bool(args.once or args.schedule)
    if needs_api and not args.api_key:
        parser.error("API key required. Set ARGILLA_API_KEY or pass --api-key")

    manager = ArgillaBackupManager(
        api_url=args.api_url,
        api_key=args.api_key or "",
        dataset_name=args.dataset,
        workspace_name=args.workspace,
        backup_dir=args.backup_dir,
        max_backups=args.max_backups,
    )

    if args.fix_existing:
        fixed = manager.fix_existing_json_encoding(args.fix_existing)
        return 0 if fixed >= 0 else 1

    if args.list:
        backups = manager.get_existing_backups()
        if not backups:
            logger.info("No backups found")
        for idx, backup in enumerate(backups, 1):
            logger.info("%d. %s", idx, backup)
        return 0

    if args.schedule:
        if not manager.connect():
            return 1
        schedule_backups(manager, args.schedule)
        return 0

    ok = manager.run_once()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
