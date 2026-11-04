"""Vectorization and canonical-expansion helpers for the shared pipeline."""

from __future__ import annotations

import numpy as np

from dite.doc.embed import (
    get_embedding_cache_variant,
    get_embedding_cache_version,
    normalize_embeddings,
)


def expand_by_file_hashes(
    canonical_indices: list[int],
    canonical_values: np.ndarray,
    file_hashes: list[str],
    total_count: int,
) -> np.ndarray:
    """Expand canonical values back to the full file list using file hashes."""
    canonical_position_by_hash = {
        file_hashes[original_index]: canonical_position
        for canonical_position, original_index in enumerate(canonical_indices)
    }
    expanded_indices = np.empty(total_count, dtype=int)
    for index, file_hash in enumerate(file_hashes):
        expanded_indices[index] = canonical_position_by_hash[file_hash]
    return canonical_values[expanded_indices]


def expand_noise_repaired_count(
    canonical_indices: list[int],
    canonical_repaired_mask: np.ndarray,
    file_hashes: list[str],
) -> int:
    """Expand canonical repaired-mask back to the full file list and count repairs."""
    if not np.any(canonical_repaired_mask):
        return 0
    expanded_repaired_mask = expand_by_file_hashes(
        canonical_indices,
        canonical_repaired_mask,
        file_hashes,
        len(file_hashes),
    )
    return int(np.sum(expanded_repaired_mask))


def expand_document_features_by_file_hashes(
    canonical_indices: list[int],
    canonical_document_features: list,
    file_hashes: list[str],
) -> list:
    """Expand canonical document features back to the full file list."""
    canonical_position_by_hash = {
        file_hashes[original_index]: canonical_position
        for canonical_position, original_index in enumerate(canonical_indices)
    }
    return [
        canonical_document_features[canonical_position_by_hash[file_hash]]
        for file_hash in file_hashes
    ]


def vectorize_files(
    *,
    files,
    file_hashes,
    contents,
    options,
    config,
    cache,
    client,
    logger,
    get_embeddings_fn,
) -> np.ndarray:
    """Build canonical embeddings with optional embedding-cache reuse."""
    if options.use_cache and options.use_embedding_cache and cache:
        embeddings_list: list[tuple[int, np.ndarray]] = []
        need_embedding_indices: list[int] = []
        need_embedding_contents: list[str] = []
        embedding_model = config.models.embedding
        cache_model_version = get_embedding_cache_version(
            embedding_model,
            options.embedding_input_mode,
            variant=get_embedding_cache_variant(config, options.embedding_input_mode),
        )

        for i, (file, content, file_hash) in enumerate(
            zip(files, contents, file_hashes, strict=False)
        ):
            cached_embedding = cache.get_embedding(
                file,
                file_hash,
                required_model_version=cache_model_version,
            )
            if cached_embedding is not None:
                embeddings_list.append((i, cached_embedding))
                continue
            need_embedding_indices.append(i)
            need_embedding_contents.append(content)

        logger.debug(
            "Embedding cache summary: "
            f"hits={len(embeddings_list)}, "
            f"misses={len(need_embedding_indices)}"
        )

        if need_embedding_contents:
            file_names = [files[i].name for i in need_embedding_indices]
            new_embeddings = get_embeddings_fn(
                client,
                need_embedding_contents,
                config=config,
                file_names=file_names,
                embedding_model=embedding_model,
                input_mode=options.embedding_input_mode,
            )

            for idx, embedding in zip(
                need_embedding_indices, new_embeddings, strict=False
            ):
                embeddings_list.append((idx, embedding))
                file = files[idx]
                file_hash = file_hashes[idx]
                cache.update_embedding(
                    file_path=file,
                    file_hash=file_hash,
                    embedding=embedding,
                    model_version=cache_model_version,
                )

        embeddings_list.sort(key=lambda item: item[0])
        return normalize_embeddings(
            np.array([embedding for _, embedding in embeddings_list])
        )

    file_names = [file.name for file in files]
    return get_embeddings_fn(
        client,
        contents,
        config=config,
        file_names=file_names,
        embedding_model=config.models.embedding,
        input_mode=options.embedding_input_mode,
    )
