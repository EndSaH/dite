"""Organize command helpers."""

from __future__ import annotations

from dite.app.i18n import t


def apply_clustering_to_preview(
    *,
    preview,
    files,
    labels,
    cluster_names,
) -> None:
    """Populate an organize preview from pipeline clustering output.

    Labels are visited in order, so when two clusters end up with the same
    folder name the lower label keeps it and the other gets the ``_2`` suffix.
    """
    for label in sorted(set(labels)):
        if label == -1:
            noise_files = [
                file
                for file, item_label in zip(files, labels, strict=False)
                if item_label == label
            ]
            preview.add_noise(noise_files)
            continue

        cluster_name = cluster_names.get(label, f"Cluster_{label}")
        cluster_files = [
            file
            for file, item_label in zip(files, labels, strict=False)
            if item_label == label
        ]
        preview.add_cluster(cluster_name, cluster_files)


def finalize_organize_run(
    *,
    preview,
    logger,
    output_script,
    execute,
) -> None:
    """Display and optionally execute an organize preview."""
    preview.display(logger.console)

    if output_script:
        preview.generate_script(output_script)
        logger.success(t("organize_script_generated", path=output_script))
        logger.print(t("organize_script_hint"))
        return

    if execute:
        logger.print("\n")
        typer = __import__("typer")
        confirm = typer.confirm(t("organize_confirm"))
        if confirm:
            from dite.flow.move import JOURNAL_DIR_NAME, find_latest_journal

            try:
                success, failed = preview.execute(
                    dry_run=False,
                    journal_dir=preview.target_folder / JOURNAL_DIR_NAME,
                )
            except KeyboardInterrupt:
                # Typer would end the run without a word: say how to undo
                # what was moved (or the folders made) before the interrupt.
                logger.error(t("error_interrupted"))
                journal_path = getattr(preview, "journal_path", None)
                if journal_path is not None and journal_path == find_latest_journal(
                    preview.target_folder
                ):
                    _print_journal_written(logger, preview, journal_path)
                raise typer.Exit(130) from None
            logger.success(t("organize_done", success=success, failed=failed))
            journal_path = getattr(preview, "journal_path", None)
            if journal_path is not None and success:
                _print_journal_written(logger, preview, journal_path)
        else:
            logger.print(t("organize_cancelled"))
        return

    logger.print(f"\n[dim]{t('organize_dry_run_hint')}[/dim]")


def _print_journal_written(logger, preview, journal_path) -> None:
    from rich.markup import escape

    logger.print(
        t(
            "organize_journal_written",
            path=escape(str(journal_path)),
            target=escape(str(preview.target_folder)),
        )
    )
