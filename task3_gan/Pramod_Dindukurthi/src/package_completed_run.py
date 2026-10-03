"""Package a completed phase without reading the active training checkpoint/logs.

Standard library only. Run on the GPU machine; no email is sent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--step', type=int,
                        help='Completed checkpoint step; default: highest completed step in this run.')
    parser.add_argument('--weights', choices=['ema', 'raw'], default='ema')
    parser.add_argument('--notebook', help='Optional path to the saved, executed lab notebook.')
    parser.add_argument('--output-dir')
    parser.add_argument('--full', action='store_true',
                        help='Include frozen checkpoint, evaluation images and all completed previews.')
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    completed = []
    for p in (run / 'reproducibility/manifests').glob('completed_*.json'):
        obj = json.loads(p.read_text())
        if isinstance(obj.get('end_step'), int):
            completed.append((p, obj))
    if not completed:
        raise RuntimeError('No completed training manifest; do not package an active phase.')
    if args.step is None:
        args.step = max(obj['end_step'] for _, obj in completed)
    selected = [(p, obj) for p, obj in completed if obj['end_step'] == args.step]
    if not selected:
        raise RuntimeError(f'No completed training manifest for step {args.step}.')
    if len(selected) != 1:
        raise RuntimeError('Multiple completed phases match this step; select a unique run directory.')
    completed_path, summary = selected[0]
    checkpoint = run / f'checkpoints/evaluation/step_{args.step:08d}.pt'
    if not checkpoint.is_file():
        raise RuntimeError('Frozen evaluation checkpoint is missing. Do not substitute latest_checkpoint.pt.')
    output = Path(args.output_dir).resolve() if args.output_dir else run.parent / 'transfer'
    if output == run or run in output.parents:
        raise ValueError('Put the archive outside the training run directory.')
    output.mkdir(parents=True, exist_ok=True)
    mode = 'full' if args.full else 'email'
    target = output / f'task3_step_{args.step:08d}_{mode}_{time.time_ns()}.zip'
    partial = target.with_suffix('.zip.partial')
    files = {}

    def add(p, alias=None):
        if p.is_file():
            name = alias or f'task3_gan/{run.name}/{p.relative_to(run).as_posix()}'
            if name in files and files[name] != p:
                raise ValueError(f'Conflicting archive path: {name}')
            files[name] = p

    history_phases = [(p, obj) for p, obj in completed
                      if obj['end_step'] <= args.step
                      and obj.get('config') == summary.get('config')
                      and obj.get('split_sha256') == summary.get('split_sha256')]
    for phase_path, phase in history_phases:
        segment = phase['segment']
        add(phase_path)
        add(run / f'reproducibility/manifests/run_{segment}.json')
        add(run / f'reproducibility/manifests/run_{segment}.requirements.txt')
        for name in (f'history_{segment}.csv', f'train_{segment}.log'):
            p = run / 'reproducibility/raw_logs' / name
            if not p.is_file():
                raise RuntimeError(f'Completed phase log is missing: {p}')
            add(p)
    for name in ('data_processed/splits.json', 'data_processed/runtime_config.yaml',
                 'README.md', 'requirements.txt', 'evaluate_local.py',
                 'results.md', 'failure_analysis.md'):
        add(run / name)
    for folder in ('src', 'configs', 'reference'):
        for p in (run / folder).rglob('*'):
            if '__pycache__' not in p.parts and p.suffix in {'.py', '.yaml', '.yml', '.ipynb', '.md'}:
                add(p)
    for p in run.glob('*.ipynb'):
        add(p)
    if args.notebook:
        notebook = Path(args.notebook).resolve()
        if not notebook.is_file() or notebook.suffix != '.ipynb':
            raise FileNotFoundError('The --notebook argument must name a saved .ipynb file.')
        add(notebook, f'task3_gan/{run.name}/notebooks/executed_{notebook.name}')
    exports = []
    manifest_hashes = {}
    local_metrics_present = False
    for folder in ('evaluations', 'official'):
        for manifest in (run / 'outputs' / folder).glob('*/export_manifest.json'):
            meta = json.loads(manifest.read_text())
            if meta.get('step') != args.step:
                continue
            if meta.get('checkpoint_sha256') != summary['checkpoint_sha256']:
                raise RuntimeError(f'Export checkpoint differs from completed phase: {manifest}')
            exports.append(str(manifest.parent.relative_to(run)))
            manifest_hashes[hashlib.sha256(manifest.read_bytes()).hexdigest()] = meta
            if meta.get('split') == 'val' and (manifest.parent / 'full_metrics_report.csv').is_file():
                local_metrics_present = True
            for p in manifest.parent.rglob('*'):
                if args.full or p.suffix in {'.json', '.csv', '.md', '.txt'}:
                    add(p)
    # Historical candidate summaries support comparison without copying old images.
    for folder in ('evaluations', 'official'):
        for manifest in (run / 'outputs' / folder).glob('*/export_manifest.json'):
            meta = json.loads(manifest.read_text())
            if not isinstance(meta.get('step'), int) or meta['step'] >= args.step:
                continue
            for p in manifest.parent.rglob('*'):
                if p.suffix in {'.json', '.csv', '.md', '.txt'}:
                    add(p)
    # Use per-candidate submissions; never collect another segment's live logs.
    submission_candidates = []
    for p in run.glob(f'submission_step_{args.step:08d}_*'):
        if p.suffix in {'.csv', '.json'}:
            add(p)
        if p.name.endswith('.metrics.json'):
            payload = json.loads(p.read_text())
            meta = manifest_hashes.get(payload.get('export_manifest_sha256'))
            csv_path = p.with_name(p.name.removesuffix('.metrics.json') + '.csv')
            if meta and meta.get('weights') == args.weights and csv_path.is_file():
                submission_candidates.append((csv_path, p, payload))
    # Preserve exact evaluator output under the instructor's canonical CSV name.
    submission_candidates.sort(key=lambda item: item[0].name)
    if submission_candidates:
        csv_path, metrics_path, payload = submission_candidates[0]
        import csv
        import math
        with csv_path.open(newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if reader.fieldnames != ['ID', 'FID', 'MiFID'] or len(rows) != 1:
                raise ValueError('Official submission schema is invalid.')
        if any(not math.isfinite(float(rows[0][key])) or
               float(rows[0][key]) != float(payload['submission'][key])
               for key in ('FID', 'MiFID')):
            raise ValueError('Submission CSV differs from its metric provenance.')
        add(csv_path, f'task3_gan/{run.name}/submission.csv')
        add(metrics_path, f'task3_gan/{run.name}/submission_provenance.json')
    draft = run / 'results_draft.md'
    if draft.is_file() and summary['checkpoint_sha256'] in draft.read_text():
        for name in ('results_draft.md', 'full_metrics_report.csv', 'metrics_report.csv'):
            add(run / name)
        for name in ('adversarial.png', 'cycle_identity.png', 'gradients.png', 'learning_rate.png'):
            add(run / 'outputs/plots' / name)
    audit = run / 'outputs/human_audit'
    audit_matches = False
    if (audit / 'private_key.json').is_file():
        key = json.loads((audit / 'private_key.json').read_text())
        audit_matches = key.get('checkpoint_sha256') == summary['checkpoint_sha256']
        if audit_matches:
            for p in audit.rglob('*'):
                if args.full or p.suffix in {'.json', '.csv', '.md', '.txt'}:
                    add(p)
    selection = run / 'selection.json'
    if selection.is_file() and json.loads(selection.read_text()).get('checkpoint_sha256') == summary['checkpoint_sha256']:
        add(selection)
    add(completed_path, f'task3_gan/{run.name}/training_summary.json')
    previews = []
    for p in (run / 'outputs/plots').glob('preview_*.png'):
        match = re.fullmatch(r'preview_(\d+)\.png', p.name)
        if match and int(match[1]) <= args.step:
            previews.append((int(match[1]), p))
    previews.sort()
    chosen = previews if args.full else previews[:1] + previews[-3:]
    for _, p in chosen:
        add(p)
    if args.full:
        add(checkpoint)
    evidence = {
        'step': args.step, 'archive_mode': mode,
        'submission_weights': args.weights,
        'completed_training_summary': summary,
        'completed_history_segments': [obj['segment'] for _, obj in history_phases],
        'frozen_checkpoint_relative_path': str(checkpoint.relative_to(run)),
        'checkpoint_binary_included': args.full,
        'evaluation_exports': exports,
        'preview_policy': 'all through completed step' if args.full else 'first and final three through completed step',
        'exclusions': ['active training logs and manifests', 'latest_checkpoint.pt',
                       'periodic checkpoint collection', 'shared original dataset',
                       'background console/status files', 'reports for a different checkpoint'],
        'notes': ['This archive describes a completed phase, not the current overnight run.',
                  'Source files are the versions present when packaging; the completed manifest records training source provenance.',
                  'Save the executed notebook and Kaggle screenshot separately if absent from this folder.',
                  'Email mode is evidence only, not a checkpoint backup. Full mode contains the frozen resumable checkpoint.'],
        'completion_status': {
            'official_submission_csv_present': bool(submission_candidates),
            'local_validation_metrics_present': local_metrics_present,
            'human_audit_summary_present': audit_matches and (audit / 'audit_summary.json').is_file(),
            'note': 'Missing rubric evidence still needs completion; creating a ZIP does not complete the report or human audit.'},
        'files': [],
    }
    try:
        with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name, p in sorted(files.items()):
                before = p.stat()
                info = zipfile.ZipInfo.from_file(p, name)
                info.compress_type = zipfile.ZIP_STORED if p.suffix in {'.pt', '.jpg', '.jpeg', '.png'} else zipfile.ZIP_DEFLATED
                digest = hashlib.sha256()
                with p.open('rb') as source, archive.open(info, 'w', force_zip64=True) as dest:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        dest.write(chunk)
                after = p.stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f'File changed while packaging; archive discarded: {p}')
                if p == checkpoint and digest.hexdigest() != summary['checkpoint_sha256']:
                    raise RuntimeError('Frozen checkpoint checksum differs from the completed manifest.')
                evidence['files'].append({'path': name, 'bytes': before.st_size, 'sha256': digest.hexdigest()})
            archive.writestr('ARCHIVE_EVIDENCE.json', json.dumps(evidence, indent=2))
            archive.writestr('PACKAGE_README.md',
                f'# Completed Task 3 snapshot\n\nCheckpoint step: {args.step}. Submission weights: {args.weights}.\n\n'
                'This package contains completed-phase evidence and excludes the active training job. '
                'The archive manifest records checksums and which rubric evidence is present.\n\n'
                'To run on another computer, restore the original dataset to '
                '`task3_gan/data/monet_jpg/` and `task3_gan/data/photo_jpg/`, install a matching '
                'CUDA PyTorch/torchvision pair and the member requirements, and update notebook paths. '
                'Keep the provided split manifest and runtime configuration unchanged when resuming.\n\n'
                'The shared original dataset is not included. Save your executed notebook and '
                'Kaggle screenshot separately, or supply `--notebook` while packaging. '
                'The written report, all requested metrics, and independent human ratings still need '
                'completion if absent. No submission or email is sent by this packager.\n')
        partial.rename(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(f'Created: {target}')
    print(f'Size: {target.stat().st_size / 1024**2:.1f} MiB; files: {len(files)}')
    print(f'Checkpoint step: {args.step}; completed history phases: {len(history_phases)}')
    print('Evidence status:', json.dumps(evidence['completion_status']))
    print('Training was not stopped. No email or upload was performed.')


if __name__ == '__main__':
    main()
