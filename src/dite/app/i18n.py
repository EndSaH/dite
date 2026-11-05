"""Internationalization (i18n) module for DITE."""

from dataclasses import dataclass
from typing import Literal

# Supported locales
Locale = Literal["zh-CN", "en"]
DEFAULT_LOCALE: Locale = "en"


@dataclass
class Messages:
    """All translatable messages in DITE."""

    # CLI general
    cli_title: str
    cli_description: str
    version_prefix: str

    # Scan command
    scan_description: str
    scan_folder_not_found: str
    scan_no_files: str
    scan_found_files: str
    scan_scanning: str
    scan_extracting: str
    scan_processing: str
    scan_extraction_done: str
    scan_vectorizing: str
    scan_vectorizing_done: str
    scan_clustering: str
    scan_clustering_done: str
    scan_naming: str
    scan_naming_done: str
    scan_report_saved: str
    scan_status_knn_suffix: str
    scan_status_small_merged_suffix: str
    scan_status_name_merged_suffix: str
    pdf_check_description: str
    pdf_check_no_pdfs: str
    pdf_check_found_pdfs: str
    pdf_check_done: str
    pdf_check_note: str
    pdf_check_weak_table_title: str
    pdf_check_verbose_table_title: str
    pdf_check_table_file: str
    pdf_check_table_primary_extractor: str
    pdf_check_table_source_profile: str
    pdf_check_table_reason: str
    pdf_check_table_selected_source: str
    pdf_check_table_effective_length: str
    pdf_check_table_source_effective_length: str
    pdf_check_table_final_effective_length: str
    pdf_check_table_fallback_needed: str
    pdf_check_table_vlm_page_calls: str
    pdf_check_table_sample_limit: str
    pdf_check_table_lengths: str
    pdf_check_table_fallback_vlm: str
    pdf_check_failed: str
    pdf_check_passed: str
    label_yes: str
    label_no: str

    # Cache messages
    cache_docling_hit: str
    cache_vlm_hit: str
    cache_vlm_fallback: str
    cache_duplicate: str
    scan_extraction_verbose: str
    extract_source_primary: str
    extract_source_vlm_cache: str
    extract_source_vlm_api: str
    extract_profile_native_text: str
    extract_profile_weak_text: str
    extract_profile_scanned_image: str
    extract_profile_mixed_pdf: str
    extract_profile_parser_timeout_or_broken: str
    pdf_check_reason_cached_vlm_available: str
    pdf_check_reason_effective_text_below_threshold: str
    pdf_check_reason_extractor_failed: str
    pdf_check_reason_glyph_noise_dominates: str
    pdf_check_reason_no_effective_text: str
    pdf_check_reason_text_with_glyph_noise: str
    pdf_check_reason_usable_text_layer: str
    pdf_check_reason_vlm_api_allowed: str
    pdf_check_reason_vlm_fallback_unavailable: str

    # Clustering results
    cluster_report_title: str
    cluster_total_files: str
    cluster_num_clusters: str
    cluster_num_noise: str
    cluster_knn_label: str
    cluster_merge_label: str
    cluster_knn_repair: str
    cluster_small_merged: str
    cluster_name_merged: str
    cluster_total_merged: str
    cluster_extraction_failed: str
    cluster_uncategorized: str
    cluster_default_name: str
    cluster_unnamed_label: str
    cluster_report_extraction_failed_marker: str

    # Organize command
    organize_description: str
    organize_specify_mode: str
    organize_mode_help: str
    organize_folder_not_found: str
    organize_preview_title: str
    organize_source_folder: str
    organize_target_folder: str
    organize_files_found: str
    organize_clusters_suggested: str
    organize_files_count: str
    organize_uncategorized: str
    organize_move_to: str
    organize_keep_in_place: str
    organize_more_files: str
    organize_script_generated: str
    organize_script_hint: str
    organize_confirm: str
    organize_done: str
    organize_journal_written: str
    undo_description: str
    cli_help_undo_path: str
    cli_help_yes: str
    cli_help_undo_dry_run: str
    undo_no_journal: str
    undo_already_done: str
    undo_preview: str
    undo_confirm: str
    undo_done: str
    undo_partial: str
    undo_passed_over: str
    undo_skipped_missing: str
    undo_skipped_conflict: str
    undo_skipped_changed: str
    assign_description: str
    cli_help_folder_assign: str
    cli_help_assign_target: str
    cli_help_min_similarity: str
    assign_no_categories: str
    assign_no_new_files: str
    assign_summary: str
    assign_decision: str
    assign_decision_unsorted: str
    organize_cancelled: str
    organize_dry_run_hint: str
    organize_script_header: str
    organize_script_usage: str
    organize_script_create_dirs: str
    organize_script_copy_files: str
    organize_script_exists: str
    organize_script_verify: str
    organize_script_done: str
    organize_script_warning: str

    # Cache commands
    cache_cleared: str
    cache_vlm_cleared: str
    cache_vacuum_failed: str
    cache_status_title: str
    cache_db_path: str
    cache_total_entries: str
    cache_with_embedding: str
    cache_current_embedding: str
    cache_stale_embedding: str
    cache_embedding_version: str
    cache_with_vlm: str
    cache_vlm_version: str
    cache_unique_hashes: str
    cache_db_size: str

    # Errors
    error_copy_failed: str
    error_api_connection_failed: str
    error_api_request_failed: str
    error_processing_failed: str
    error_interrupted: str
    error_docling_pdf_models_missing: str
    error_docling_pdf_timeout: str
    error_docling_pdf_subprocess_failed: str
    error_pdf_render_failed: str
    error_pdf_vlm_no_usable_content: str
    error_text_decode_failed: str
    error_vlm_client_not_initialized: str

    # Progress
    progress_executing: str
    progress_document_conversion: str
    progress_vlm_fallback: str
    progress_vlm_api_call: str

    # CLI help texts
    cli_help_verbose: str
    cli_help_quiet: str
    cli_help_color: str
    cli_help_version: str
    cli_help_folder_scan: str
    cli_help_folder_pdf_check: str
    cli_help_output_report: str
    cli_help_disable_cache: str
    cli_help_cached_vlm_only: str
    cli_help_disable_knn_repair: str
    cli_help_folder_organize: str
    cli_help_target_folder: str
    cli_help_preview_mode: str
    cli_help_execute_move: str
    cli_help_output_script: str
    cli_help_cache_group: str
    cli_help_cache_clear: str
    cli_help_cache_clear_vlm: str
    cli_help_cache_status: str
    cli_help_setup_group: str
    cli_help_setup_docling_pdf: str
    cli_help_setup_docling_pdf_force: str
    cli_help_setup_docling_pdf_progress: str

    # Setup command
    setup_docling_pdf_start: str
    setup_docling_pdf_done: str
    setup_docling_pdf_failed: str
    setup_docling_pdf_incomplete: str

    # Logging
    log_debug_prefix: str
    log_info_prefix: str
    log_warning_prefix: str
    log_error_prefix: str
    log_success_prefix: str

    # Debug messages
    debug_duplicate_groups: str
    debug_duplicate_group_hash: str
    debug_duplicate_group_file: str
    debug_scan_folder: str
    debug_scan_recursive: str
    debug_scan_extensions: str
    debug_scan_excluded_dirs: str
    debug_scan_no_extension_label: str
    debug_scan_skipped_unsupported: str
    debug_scan_skipped_extension_count: str
    debug_scan_skipped_ignored: str
    debug_scan_summary: str
    debug_extract_processing_file: str
    debug_extract_hash: str
    debug_extract_doc_cache_hit: str
    debug_extract_doc_cache_duplicate_source: str
    debug_extract_doc_cache_miss: str
    debug_extract_doc_result: str
    debug_extract_vlm_check: str
    debug_pdf_profile: str
    debug_extract_vlm_cache_hit: str
    debug_extract_vlm_api_call: str
    debug_extract_vlm_result: str
    debug_extract_vlm_selected: str
    debug_extract_vlm_skipped: str
    debug_extract_truncated: str
    debug_extract_summary: str
    debug_vector_cache_summary: str
    debug_vectorizing_documents: str
    debug_vectorizing_model: str
    debug_vector_fallback_names: str
    debug_vector_text_stats: str
    debug_vector_dimension: str
    debug_vector_api_usage: str
    debug_vector_batch_split: str
    debug_cluster_hdbscan_header: str
    debug_cluster_hdbscan_min_cluster_size: str
    debug_cluster_hdbscan_min_samples: str
    debug_cluster_hdbscan_epsilon: str
    debug_cluster_hdbscan_method: str
    debug_cluster_input_vectors: str
    debug_cluster_initial_result: str
    debug_cluster_sizes: str
    debug_cluster_knn_dynamic_threshold: str
    debug_cluster_knn_fixed_threshold: str
    debug_cluster_knn_assignment: str
    debug_cluster_knn_kept: str
    debug_cluster_knn_summary: str
    debug_cluster_small_merge_event: str
    debug_cluster_small_merge_skipped: str
    debug_cluster_small_merge_summary: str
    debug_cluster_name_empty_response: str
    debug_cluster_name_retry: str
    debug_cluster_name_empty_fallback: str
    debug_cluster_name_invalid_fallback: str
    debug_cluster_name_cut_off_fallback: str
    debug_cluster_name_failed_fallback: str
    debug_cluster_merge: str
    debug_cluster_name_result: str
    debug_cluster_name_async_request_failed: str
    debug_cluster_name_async_response_empty: str
    debug_cluster_name_async_response_invalid: str
    warning_cluster_naming_fallback: str
    warning_cluster_naming_first_error: str
    warning_cluster_naming_reasoning_hint: str
    debug_analyzer_json_parse_failed: str
    debug_analyzer_api_failed: str
    warning_unsupported_file_format: str
    warning_extractor_failed: str
    warning_extract_file_failed: str
    warning_extract_cache_failed: str
    warning_pdf2image_missing: str
    warning_vlm_fallback_failed: str
    warning_analyzer_default_used: str
    warning_analyzer_failed: str
    debug_vlm_page_processing: str
    debug_vlm_page_resized: str
    debug_vlm_image_size: str
    debug_vlm_api_call: str
    debug_vlm_page_done: str
    debug_vlm_page_failed: str
    debug_vlm_page_result_done: str
    debug_vlm_page_result_failed: str
    debug_vlm_page_result_empty: str
    debug_vlm_batch_summary: str
    debug_async_runtime_limits: str


# Chinese translations (default)
ZH_CN = Messages(
    # CLI general
    cli_title="DITE - 多模态文件智能聚类工具",
    cli_description=(
        "DITE - Document Insight & Taxonomy Engine\n\n多模态文件智能聚类工具"
    ),
    version_prefix="DITE version",
    # Scan command
    scan_description="扫描文件夹并进行聚类分析",
    scan_folder_not_found="文件夹 {folder} 不存在",
    scan_no_files="未找到支持的文件",
    scan_found_files="找到 {count} 个文件",
    scan_scanning="扫描文件...",
    scan_extracting="提取内容...",
    scan_processing="处理: {name}",
    scan_extraction_done="内容提取完成",
    scan_vectorizing="向量化中...",
    scan_vectorizing_done="向量化完成 (维度: {dim})",
    scan_clustering="聚类分析...",
    scan_clustering_done="聚类完成 (发现 {clusters} 个簇, {noise} 个未分类)",
    scan_naming="生成簇名称...",
    scan_naming_done="簇命名完成",
    scan_report_saved="报告已保存: {path}",
    scan_status_knn_suffix="噪音修复: {count}",
    scan_status_small_merged_suffix="小簇合并: {count}",
    scan_status_name_merged_suffix="同名合并: {count}",
    pdf_check_description="快速检查 PDF 最终提取结果是否足够可用，不做全文完整性审计",
    pdf_check_no_pdfs="未找到 PDF 文件",
    pdf_check_found_pdfs="找到 {count} 个 PDF 文件",
    pdf_check_done=(
        "PDF 快速检查完成 (文档缓存: {doc_cache_hits}, VLM缓存: {vlm_cache_hits}, "
        "主解析失败: {primary_failures}, 源内容需回退: {source_fallback_needed}, "
        "最终使用 VLM: {selected_vlm_files}, VLM页级调用: {vlm_api_page_calls}, "
        "重复文件: {duplicates}, "
        "弱内容: {weak}, 空内容: {empty})"
    ),
    pdf_check_note=(
        "VLM 仅采样前 10 页；本命令验证的是最终提取结果是否足够可用，"
        "不代表整本文档已完整提取。"
    ),
    pdf_check_weak_table_title="最终弱内容 PDF",
    pdf_check_verbose_table_title="提取明细",
    pdf_check_table_file="文件",
    pdf_check_table_primary_extractor="主解析器",
    pdf_check_table_source_profile="源内容类型",
    pdf_check_table_reason="原因",
    pdf_check_table_selected_source="最终来源",
    pdf_check_table_effective_length="有效长度",
    pdf_check_table_source_effective_length="源有效长度",
    pdf_check_table_final_effective_length="最终有效长度",
    pdf_check_table_fallback_needed="需要回退",
    pdf_check_table_vlm_page_calls="VLM页调用",
    pdf_check_table_sample_limit="采样上限",
    pdf_check_table_lengths="源->最终",
    pdf_check_table_fallback_vlm="回退; VLM页",
    pdf_check_failed="{count} 个 PDF 的最终提取结果低于阈值",
    pdf_check_passed="{count} 个 PDF 的最终提取结果通过快速检查",
    label_yes="是",
    label_no="否",
    # Cache messages
    cache_docling_hit="文档缓存: {count}",
    cache_vlm_hit="VLM缓存: {count}",
    cache_vlm_fallback="最终使用 VLM: {count}",
    cache_duplicate="重复文件: {count}",
    scan_extraction_verbose=(
        "提取细节: 主解析失败={primary_failures}, "
        "源内容需回退={source_fallback_needed}, "
        "VLM页级调用={vlm_api_page_calls}"
    ),
    extract_source_primary="主解析结果",
    extract_source_vlm_cache="VLM缓存",
    extract_source_vlm_api="VLM采样",
    extract_profile_native_text="文本层可用",
    extract_profile_weak_text="文本层过弱",
    extract_profile_scanned_image="扫描件/无文本层",
    extract_profile_mixed_pdf="正文夹杂噪音字形",
    extract_profile_parser_timeout_or_broken="主解析失败",
    pdf_check_reason_cached_vlm_available="命中 VLM 缓存",
    pdf_check_reason_effective_text_below_threshold="有效文本低于阈值",
    pdf_check_reason_extractor_failed="主解析器失败",
    pdf_check_reason_glyph_noise_dominates="噪音字形占主导",
    pdf_check_reason_no_effective_text="没有有效文本",
    pdf_check_reason_text_with_glyph_noise="文本夹杂噪音字形",
    pdf_check_reason_usable_text_layer="文本层可用",
    pdf_check_reason_vlm_api_allowed="允许 VLM 采样",
    pdf_check_reason_vlm_fallback_unavailable="VLM 回退不可用",
    # Clustering results
    cluster_report_title="=== 聚类分析报告 ===",
    cluster_total_files="总文件数:",
    cluster_num_clusters="发现簇数:",
    cluster_num_noise="未分类文件:",
    cluster_knn_label="噪音修复:",
    cluster_merge_label="合并:",
    cluster_knn_repair="{count} 个噪音点已归类",
    cluster_small_merged="{count} 个小簇已合并",
    cluster_name_merged="{count} 个同名簇已合并",
    cluster_total_merged="总计合并 {count} 个簇",
    cluster_extraction_failed="内容提取失败: {count}",
    cluster_uncategorized="未分类",
    cluster_default_name="簇_{label}",
    cluster_unnamed_label="未命名",
    cluster_report_extraction_failed_marker="(提取失败)",
    # Organize command
    organize_description="整理文件夹中的文档",
    organize_specify_mode="请指定 --dry-run, --execute 或 --output-script",
    organize_mode_help=(
        "使用 --dry-run 预览，使用 --execute 执行，使用 --output-script 生成脚本"
    ),
    organize_folder_not_found="文件夹 {folder} 不存在",
    organize_preview_title="📁 DITE 文件整理预览",
    organize_source_folder="源文件夹:",
    organize_target_folder="目标文件夹:",
    organize_files_found="发现文件:",
    organize_clusters_suggested="建议簇数:",
    organize_files_count="{count} 个文件",
    organize_uncategorized="未分类",
    organize_move_to="→ 移动到: ./{name}/",
    organize_keep_in_place="→ 保留在原位置",
    organize_more_files="...（还有 {count} 个）",
    organize_script_generated="脚本已生成: {path}",
    organize_script_hint="请检查脚本内容后执行: chmod +x && ./organize.sh",
    organize_confirm="确认执行文件移动?",
    organize_done="完成: {success} 个文件已移动, {failed} 个失败",
    organize_journal_written="移动记录已写入: {path}（撤销: dite undo {target}）",
    undo_description="撤销一次 organize/assign --execute 的文件移动",
    cli_help_undo_path=(
        "移动记录文件，或目标文件夹（使用其 .dite/ 下最新一份未撤销的记录）"
    ),
    cli_help_yes="不再询问确认",
    cli_help_undo_dry_run="只预览要移回的文件，不实际移动",
    undo_no_journal="找不到可撤销的移动记录: {path}",
    undo_already_done="这份记录已经撤销过: {path}",
    undo_preview="将把 {count} 个文件移回原位（记录: {path}）",
    undo_confirm="确认撤销这些移动?",
    undo_done="撤销完成: {restored} 个已移回, {skipped} 个跳过",
    undo_partial=(
        "{count} 个文件没有移回，这份记录保持未撤销；处理好上面的问题后"
        "再次运行 dite undo {path} 即可重试"
    ),
    undo_passed_over=(
        "已跳过记录 {path}：其中的文件现在都无法移回（已不存在或已被修改）。"
        "这份记录保持未撤销，文件回来后（例如改名的文件夹改回原名）"
        "再次运行 dite undo 会重新找到它"
    ),
    undo_skipped_missing="跳过（整理后的文件已不存在）: {path}",
    undo_skipped_conflict="跳过（原位置已有同名文件）: {path}",
    undo_skipped_changed=(
        "跳过（整理后的文件已被修改或替换，不是当初移过去的那个；"
        "它留在当前位置，需要的话请手动移回）: {path}"
    ),
    assign_description="把新文件归入已有的子文件夹",
    cli_help_folder_assign="包含新文件的文件夹",
    cli_help_assign_target=(
        "已有分类子文件夹所在的目录（默认就是 FOLDER，此时只处理 FOLDER "
        "根目录下的散落文件）"
    ),
    cli_help_min_similarity=(
        "最低余弦相似度；不设时由每个文件夹自身成员的相似度分布决定门槛"
    ),
    assign_no_categories=(
        "{target} 下没有可用的子文件夹（至少需要一个含受支持文件的子文件夹）"
    ),
    assign_no_new_files="没有需要归档的新文件",
    assign_summary=(
        "{assigned} 个文件可以归入已有文件夹，{unsorted} 个不够接近、保持原位"
    ),
    assign_decision="{file} → {folder}（相似度 {similarity:.3f}，门槛 {floor:.3f}）",
    assign_decision_unsorted=(
        "{file} 保持原位（最接近 {folder}，相似度 {similarity:.3f}，门槛 "
        "{floor}）"
    ),
    organize_cancelled="已取消",
    organize_dry_run_hint=(
        "这是预览模式。使用 --execute 执行或 --output-script 生成脚本。"
    ),
    organize_script_header="DITE 文件整理脚本",
    organize_script_usage="使用方法:",
    organize_script_create_dirs="创建目录",
    organize_script_copy_files="复制文件（使用 cp + rm 而非 mv，更安全）",
    organize_script_exists="目标位置已有文件，为避免覆盖已停止:",
    organize_script_verify="验证复制成功后删除原文件",
    organize_script_done="✅ 完成！共复制 {count} 个文件",
    organize_script_warning="⚠️  原文件未删除，请手动确认后执行删除",
    # Cache commands
    cache_cleared="已清除 {count} 条缓存记录",
    cache_vlm_cleared=(
        "已清除 {count} 条 VLM 缓存，并使 {embeddings} 条基于它的 Embedding 失效"
        "（文档转换结果和其他 Embedding 已保留）"
    ),
    cache_vacuum_failed=(
        "无法压缩缓存文件（{error}）。已淘汰条目的空间会被复用，但文件会保持在 "
        "cache.max_size_gb 以上，直到之后某次压缩成功。可释放磁盘空间或运行 "
        "`dite cache clear`。"
    ),
    cache_status_title="缓存状态",
    cache_db_path="数据库路径:",
    cache_total_entries="总条目数:",
    cache_with_embedding="含 Embedding:",
    cache_current_embedding="当前 Embedding:",
    cache_stale_embedding="过期 Embedding:",
    cache_embedding_version="当前 Embedding 缓存版本:",
    cache_with_vlm="含 VLM 回退:",
    cache_vlm_version="VLM 缓存版本:",
    cache_unique_hashes="唯一哈希数:",
    cache_db_size="数据库大小:",
    # Errors
    error_copy_failed="复制验证失败",
    error_api_connection_failed=(
        "API 连接失败。请检查 api.base_url、api.api_key、网络和代理设置。"
    ),
    error_api_request_failed="API 请求失败: {error}",
    error_processing_failed="处理失败: {error}",
    error_interrupted="已中断，本次运行已停止。",
    error_docling_pdf_models_missing="Docling PDF 模型未安装。请先运行 `{command}`。",
    error_docling_pdf_timeout="Docling PDF 提取超时: {seconds:g}s",
    error_docling_pdf_subprocess_failed=(
        "Docling PDF 子进程未返回结果就退出了（退出码 {exitcode}）"
    ),
    error_pdf_render_failed="PDF 渲染失败",
    error_pdf_vlm_no_usable_content="VLM 回退未返回任何可用内容",
    error_text_decode_failed="无法解码文件，尝试的编码: {encodings}",
    error_vlm_client_not_initialized="VLM 客户端未初始化",
    # Progress
    progress_executing="执行",
    progress_document_conversion="执行文档转换...",
    progress_vlm_fallback="需要 VLM 回退 (有效内容不足)",
    progress_vlm_api_call="调用 VLM API...",
    # CLI help texts
    cli_help_verbose="详细输出",
    cli_help_quiet="静默模式",
    cli_help_color="强制启用颜色输出",
    cli_help_version="显示版本",
    cli_help_folder_scan="要扫描的文件夹",
    cli_help_folder_pdf_check="要检查的 PDF 文件夹",
    cli_help_output_report="输出 JSON 报告路径",
    cli_help_disable_cache="禁用缓存",
    cli_help_cached_vlm_only="只使用已缓存的 VLM 结果，不调用 VLM API",
    cli_help_disable_knn_repair="禁用噪音修复",
    cli_help_folder_organize="要整理的文件夹",
    cli_help_target_folder="目标文件夹",
    cli_help_preview_mode="预览模式",
    cli_help_execute_move="执行文件移动",
    cli_help_output_script="输出 shell 脚本路径",
    cli_help_cache_group="缓存管理",
    cli_help_cache_clear="清除所有缓存",
    cli_help_cache_clear_vlm="仅清除 VLM 缓存",
    cli_help_cache_status="查看缓存状态",
    cli_help_setup_group="环境准备",
    cli_help_setup_docling_pdf="安装 DITE 所需的本地 Docling PDF 模型",
    cli_help_setup_docling_pdf_force="强制重新下载模型",
    cli_help_setup_docling_pdf_progress="显示下载进度",
    # Setup command
    setup_docling_pdf_start="开始安装 Docling PDF 模型: {path}",
    setup_docling_pdf_done="Docling PDF 模型已就绪: {path}",
    setup_docling_pdf_failed="安装 Docling PDF 模型失败: {error}",
    setup_docling_pdf_incomplete="Docling PDF 模型不完整: {path}",
    # Logging
    log_debug_prefix="调试:",
    log_info_prefix="信息:",
    log_warning_prefix="警告:",
    log_error_prefix="错误:",
    log_success_prefix="成功:",
    # Debug messages
    debug_duplicate_groups="发现重复文件组:",
    debug_duplicate_group_hash="  [hash]{hash}...[/hash]",
    debug_duplicate_group_file="    - [path]{name}[/path]",
    debug_scan_folder="扫描目录: [path]{folder}[/path]",
    debug_scan_recursive="递归扫描: {recursive}",
    debug_scan_extensions="支持的扩展名: {extensions}",
    debug_scan_excluded_dirs="排除目录: {paths}",
    debug_scan_no_extension_label="(无扩展名)",
    debug_scan_skipped_unsupported="跳过 {count} 个不支持的文件",
    debug_scan_skipped_extension_count="  {extension}: {count} 个",
    debug_scan_skipped_ignored="跳过 {dirs} 个忽略目录和 {files} 个忽略文件",
    debug_scan_summary=(
        "扫描完成: 支持文件 {supported} 个, "
        "排除目录内文件 {excluded} 个, 不支持文件 {skipped} 个"
    ),
    debug_extract_processing_file="处理文件: [path]{path}[/path]",
    debug_extract_hash="  文件哈希: [hash]{hash}[/hash]",
    debug_extract_doc_cache_hit="  文档缓存命中",
    debug_extract_doc_cache_duplicate_source=(
        "    复用已缓存源文件: [path]{source}[/path]"
    ),
    debug_extract_doc_cache_miss="  文档缓存未命中",
    debug_extract_doc_result=(
        "  文档提取: extractor={extractor}, success={success}, "
        "length={length}, error={error}"
    ),
    debug_extract_vlm_check=(
        "  VLM 检查: suffix={suffix}, effective_length={effective_length}, "
        "threshold={threshold}, needed={needed}"
    ),
    debug_pdf_profile=(
        "  PDF 分类: kind={kind}, reason={reason}, "
        "effective_length={effective_length}, "
        "glyph_noise={glyph_noise_tokens}, fallback={needs_vlm_fallback}"
    ),
    debug_extract_vlm_cache_hit="  VLM 缓存命中 (length={length})",
    debug_extract_vlm_api_call="  调用 VLM 回退",
    debug_extract_vlm_result=(
        "  VLM 结果: success={success}, length={length}, error={error}"
    ),
    debug_extract_vlm_selected="  采用 VLM 结果 ({vlm_length} > {doc_length})",
    debug_extract_vlm_skipped="  保留文档提取结果 ({doc_length} >= {vlm_length})",
    debug_extract_truncated="  内容截断: {original} -> {limit}",
    debug_extract_summary=(
        "提取汇总: doc缓存命中={doc_cache_hits}, VLM缓存命中={vlm_cache_hits}, "
        "主解析失败={primary_failures}, 源内容需回退={source_fallback_needed}, "
        "最终使用VLM={selected_vlm_files}, VLM页级调用={vlm_api_page_calls}, "
        "重复文件={duplicates}"
    ),
    debug_vector_cache_summary=("Embedding 缓存: 命中 {hits} 个, 未命中 {misses} 个"),
    debug_vectorizing_documents="向量化 {count} 个文档",
    debug_vectorizing_model="使用模型: {model}",
    debug_vector_fallback_names="  {count} 个文件使用文件名回退: {names}",
    debug_vector_text_stats=(
        "  文本长度: 最小={min_length}, 最大={max_length}, 平均={avg_length}"
    ),
    debug_vector_dimension="  向量维度: {dimension}",
    debug_vector_api_usage="  API 用量: {tokens} tokens",
    debug_vector_batch_split=(
        "  embedding 批次被拒绝（status={status}），将 {size} 条拆成两半重试"
    ),
    debug_cluster_hdbscan_header="HDBSCAN 聚类参数:",
    debug_cluster_hdbscan_min_cluster_size="  min_cluster_size: {value}",
    debug_cluster_hdbscan_min_samples="  min_samples: {value}",
    debug_cluster_hdbscan_epsilon="  cluster_selection_epsilon: {value}",
    debug_cluster_hdbscan_method="  cluster_selection_method: {value}",
    debug_cluster_input_vectors="  输入向量: {count} 个, {dimension} 维",
    debug_cluster_initial_result="初始聚类结果: {clusters} 个簇, {noise} 个噪音点",
    debug_cluster_sizes="  簇大小: {sizes}",
    debug_cluster_knn_dynamic_threshold=(
        "k-NN 动态阈值: {threshold:.4f} (mean_core_distance={mean_core_distance:.4f})"
    ),
    debug_cluster_knn_fixed_threshold="k-NN 固定阈值: {threshold:.4f}",
    debug_cluster_knn_assignment=(
        "  噪音点 {index} ({name}) -> 簇 {label} (distance={distance:.4f}, "
        "threshold={threshold:.4f})"
    ),
    debug_cluster_knn_kept=(
        "  噪音点 {index} ({name}) 保持未分类，候选簇 {label} "
        "(distance={distance:.4f}, "
        "threshold={threshold:.4f})"
    ),
    debug_cluster_knn_summary=("k-NN 修复了 {repaired} 个噪音点，保留 {kept} 个噪音点"),
    debug_cluster_small_merge_event=(
        "  小簇 {source} (size={source_size}) -> 簇 {target} "
        "(target_size={target_size}, similarity={similarity:.4f})"
    ),
    debug_cluster_small_merge_skipped=(
        "  小簇 {source} (size={source_size}) 跳过，最佳目标 {target} "
        "(target_size={target_size}, similarity={similarity:.4f}, reason={reason})"
    ),
    debug_cluster_small_merge_summary=(
        "小簇再合并: candidates={candidates}, merged={merged}, skipped={skipped}"
    ),
    debug_cluster_name_empty_response=(
        "簇命名响应为空: model={model}, finish_reason={finish_reason}, "
        "reasoning_chars={reasoning_chars}"
    ),
    debug_cluster_name_retry="簇命名请求失败 (尝试 {attempt}/{max_retries})：{error}",
    debug_cluster_name_empty_fallback="簇命名返回空结果，使用回退名称: {fallback}",
    debug_cluster_name_invalid_fallback="簇命名返回无效名称，使用回退名称: {fallback}",
    debug_cluster_name_cut_off_fallback=(
        "簇命名回复在第一行结束前被 max_tokens 截断，使用回退名称: {fallback}"
    ),
    debug_cluster_name_failed_fallback="簇命名失败，使用回退名称 {fallback}: {error}",
    debug_cluster_merge="合并簇 {source} -> {target} (名称: {name})",
    debug_cluster_name_result="簇 {label} 命名为 {name} (files={count})",
    debug_cluster_name_async_request_failed=(
        "簇 {label} 异步命名请求失败: error={error}, wait={wait_sec:.3f}s, "
        "request={request_sec:.3f}s, fallback={fallback}"
    ),
    debug_cluster_name_async_response_empty=(
        "簇 {label} 异步命名结果为空: wait={wait_sec:.3f}s, "
        "request={request_sec:.3f}s, fallback={fallback}"
    ),
    debug_cluster_name_async_response_invalid=(
        "簇 {label} 异步命名结果非法: wait={wait_sec:.3f}s, "
        "request={request_sec:.3f}s, fallback={fallback}"
    ),
    warning_cluster_naming_fallback=(
        "LLM 簇命名有 {fallback}/{total} 个簇退回了启发式名称（请求失败 {failed} 个，"
        "回复为空、只有推理内容或被截断 {empty} 个，名称不可用 {invalid} 个）。"
        "使用 --verbose 查看每个簇的详情。"
    ),
    warning_cluster_naming_first_error="第一个簇命名错误：{error}",
    warning_cluster_naming_reasoning_hint=(
        "推理过程可能占满了命名输出：请保持 "
        "request_profiles.cluster_naming.reasoning_mode: off，"
        "按服务商设置 api.reasoning_param_style（见 docs/configuration.md），"
        "或调大其 max_tokens。"
    ),
    debug_analyzer_json_parse_failed=(
        "JSON 解析失败 (尝试 {attempt}/{max_retries}): {error}"
    ),
    debug_analyzer_api_failed="API 调用失败 (尝试 {attempt}/{max_retries}): {error}",
    warning_unsupported_file_format="不支持的文件格式: {suffix}",
    warning_extractor_failed="{extractor} 处理 {name} 失败: {error}",
    warning_extract_file_failed="提取 {path} 失败，继续处理其他文件: {error}",
    warning_extract_cache_failed="读写 {path} 的缓存失败，提取结果照常保留: {error}",
    warning_pdf2image_missing="pdf2image 未安装，无法使用 VLM 回退",
    warning_vlm_fallback_failed="VLM 回退处理失败: {error}",
    warning_analyzer_default_used="文档分析失败，使用默认值",
    warning_analyzer_failed="文档分析失败: {error}",
    debug_vlm_page_processing="  处理第 {page}/{total} 页 ({width}x{height})",
    debug_vlm_page_resized=(
        "  缩放: {old_width}x{old_height} -> {new_width}x{new_height}"
    ),
    debug_vlm_image_size="  图片大小: {size_kb:.1f} KB",
    debug_vlm_api_call="  调用 VLM API...",
    debug_vlm_page_done="  第 {page} 页完成 ({length} 字符)",
    debug_vlm_page_failed="  第 {page} 页失败: {error}",
    debug_vlm_page_result_done=(
        "  第 {page} 页结果: success, length={length}, "
        "wait={wait_sec:.3f}s, request={request_sec:.3f}s"
    ),
    debug_vlm_page_result_failed=(
        "  第 {page} 页结果: failed, error={error}, "
        "wait={wait_sec:.3f}s, request={request_sec:.3f}s"
    ),
    debug_vlm_page_result_empty=(
        "  第 {page} 页结果: empty, wait={wait_sec:.3f}s, "
        "request={request_sec:.3f}s"
    ),
    debug_vlm_batch_summary=(
        "  VLM 批次汇总: success={success}, failed={failed}, "
        "max_wait={max_wait_sec:.3f}s, max_request={max_request_sec:.3f}s"
    ),
    debug_async_runtime_limits=(
        "异步 API 运行时限制: cluster={cluster_workers}, vlm={vlm_workers}, "
        "vlm_pages_per_document={vlm_pages_per_document}, "
        "max_connections={max_connections}"
    ),
)

# English translations
EN = Messages(
    # CLI general
    cli_title="DITE - Multimodal Document Clustering Tool",
    cli_description=(
        "DITE - Document Insight & Taxonomy Engine\n\n"
        "Multimodal document intelligent clustering tool"
    ),
    version_prefix="DITE version",
    # Scan command
    scan_description="Scan folder and perform cluster analysis",
    scan_folder_not_found="Folder {folder} does not exist",
    scan_no_files="No supported files found",
    scan_found_files="Found {count} files",
    scan_scanning="Scanning files...",
    scan_extracting="Extracting content...",
    scan_processing="Processing: {name}",
    scan_extraction_done="Content extraction completed",
    scan_vectorizing="Vectorizing...",
    scan_vectorizing_done="Vectorization completed (dim: {dim})",
    scan_clustering="Clustering...",
    scan_clustering_done=(
        "Clustering completed ({clusters} clusters, {noise} uncategorized)"
    ),
    scan_naming="Generating cluster names...",
    scan_naming_done="Cluster naming completed",
    scan_report_saved="Report saved: {path}",
    scan_status_knn_suffix="Noise repaired: {count}",
    scan_status_small_merged_suffix="Small-cluster merged: {count}",
    scan_status_name_merged_suffix="Same-name merged: {count}",
    pdf_check_description=(
        "Quickly check whether final PDF extraction output is usable; "
        "this is not a full-document completeness audit"
    ),
    pdf_check_no_pdfs="No PDF files found",
    pdf_check_found_pdfs="Found {count} PDF files",
    pdf_check_done=(
        "PDF smoke check completed (doc cache: {doc_cache_hits}, "
        "VLM cache: {vlm_cache_hits}, primary failures: {primary_failures}, "
        "fallback needed: {source_fallback_needed}, "
        "selected VLM: {selected_vlm_files}, "
        "VLM page calls: {vlm_api_page_calls}, "
        "duplicates: {duplicates}, weak: {weak}, empty: {empty})"
    ),
    pdf_check_note=(
        "VLM samples only the first 10 pages. This command checks whether the "
        "final extraction output is usable, not whether the full document was "
        "completely extracted."
    ),
    pdf_check_weak_table_title="Weak final PDF outputs",
    pdf_check_verbose_table_title="Extraction details",
    pdf_check_table_file="File",
    pdf_check_table_primary_extractor="Primary extractor",
    pdf_check_table_source_profile="Source profile",
    pdf_check_table_reason="Reason",
    pdf_check_table_selected_source="Selected source",
    pdf_check_table_effective_length="Effective length",
    pdf_check_table_source_effective_length="Source effective length",
    pdf_check_table_final_effective_length="Final effective length",
    pdf_check_table_fallback_needed="Fallback needed",
    pdf_check_table_vlm_page_calls="VLM page calls",
    pdf_check_table_sample_limit="Sample limit",
    pdf_check_table_lengths="Source->final",
    pdf_check_table_fallback_vlm="Fallback; VLM pages",
    pdf_check_failed="{count} final PDF extraction outputs are below threshold",
    pdf_check_passed="{count} PDF outputs passed the smoke check",
    label_yes="yes",
    label_no="no",
    # Cache messages
    cache_docling_hit="Doc cache: {count}",
    cache_vlm_hit="VLM cache: {count}",
    cache_vlm_fallback="Selected VLM: {count}",
    cache_duplicate="Duplicates: {count}",
    scan_extraction_verbose=(
        "Extraction details: primary_failures={primary_failures}, "
        "fallback_needed={source_fallback_needed}, vlm_page_calls={vlm_api_page_calls}"
    ),
    extract_source_primary="Primary output",
    extract_source_vlm_cache="VLM cache",
    extract_source_vlm_api="VLM sampling",
    extract_profile_native_text="Usable text layer",
    extract_profile_weak_text="Weak text layer",
    extract_profile_scanned_image="Scanned/no text layer",
    extract_profile_mixed_pdf="Text with glyph noise",
    extract_profile_parser_timeout_or_broken="Primary parser failed",
    pdf_check_reason_cached_vlm_available="VLM cache hit",
    pdf_check_reason_effective_text_below_threshold="Effective text below threshold",
    pdf_check_reason_extractor_failed="Primary extractor failed",
    pdf_check_reason_glyph_noise_dominates="Glyph noise dominates",
    pdf_check_reason_no_effective_text="No effective text",
    pdf_check_reason_text_with_glyph_noise="Text contains glyph noise",
    pdf_check_reason_usable_text_layer="Usable text layer",
    pdf_check_reason_vlm_api_allowed="VLM sampling allowed",
    pdf_check_reason_vlm_fallback_unavailable="VLM fallback unavailable",
    # Clustering results
    cluster_report_title="=== Cluster Analysis Report ===",
    cluster_total_files="Total files:",
    cluster_num_clusters="Clusters found:",
    cluster_num_noise="Uncategorized:",
    cluster_knn_label="Noise repair:",
    cluster_merge_label="Merge:",
    cluster_knn_repair="{count} noise points repaired",
    cluster_small_merged="{count} small clusters merged",
    cluster_name_merged="{count} same-name clusters merged",
    cluster_total_merged="{count} clusters merged in total",
    cluster_extraction_failed="Extraction failed: {count}",
    cluster_uncategorized="Uncategorized",
    cluster_default_name="Cluster_{label}",
    cluster_unnamed_label="Unnamed",
    cluster_report_extraction_failed_marker="(extraction failed)",
    # Organize command
    organize_description="Organize documents in folder",
    organize_specify_mode="Please specify --dry-run, --execute, or --output-script",
    organize_mode_help=(
        "Use --dry-run to preview, --execute to run, --output-script to generate script"
    ),
    organize_folder_not_found="Folder {folder} does not exist",
    organize_preview_title="📁 DITE File Organization Preview",
    organize_source_folder="Source folder:",
    organize_target_folder="Target folder:",
    organize_files_found="Files found:",
    organize_clusters_suggested="Clusters suggested:",
    organize_files_count="{count} files",
    organize_uncategorized="Uncategorized",
    organize_move_to="→ Move to: ./{name}/",
    organize_keep_in_place="→ Keep in place",
    organize_more_files="... ({count} more)",
    organize_script_generated="Script generated: {path}",
    organize_script_hint="Review the script and run: chmod +x && ./organize.sh",
    organize_confirm="Confirm file move?",
    organize_done="Done: {success} files moved, {failed} failed",
    organize_journal_written=(
        "Move journal written: {path} (undo with: dite undo {target})"
    ),
    undo_description="Undo the file moves of an organize/assign --execute run",
    cli_help_undo_path=(
        "Journal file, or a target folder (uses its latest journal in .dite/ "
        "that is not undone yet)"
    ),
    cli_help_yes="Do not ask for confirmation",
    cli_help_undo_dry_run="Only list the files that would be moved back",
    undo_no_journal="No move journal to undo at: {path}",
    undo_already_done="This journal was already undone: {path}",
    undo_preview="{count} files will be moved back (journal: {path})",
    undo_confirm="Undo these moves?",
    undo_done="Undo finished: {restored} restored, {skipped} skipped",
    undo_partial=(
        "{count} files were not moved back, so the journal stays open; fix "
        "the problems above and run dite undo {path} again to retry them"
    ),
    undo_passed_over=(
        "Passed over the journal {path}: none of its files can be moved back "
        "now (they are missing or were changed). It stays open, so dite undo "
        "finds it again once they are back (a renamed folder renamed back, say)"
    ),
    undo_skipped_missing="Skipped (the moved file no longer exists): {path}",
    undo_skipped_conflict=(
        "Skipped (a file already exists at the original location): {path}"
    ),
    undo_skipped_changed=(
        "Skipped (the moved file was changed or replaced since the move; it "
        "stays where it is, move it back by hand if you want): {path}"
    ),
    assign_description="Sort new files into existing subfolders",
    cli_help_folder_assign="Folder that contains the new files",
    cli_help_assign_target=(
        "Folder whose subfolders are the existing categories (default: FOLDER "
        "itself; then only loose files at its top level are sorted)"
    ),
    cli_help_min_similarity=(
        "Minimum cosine similarity; by default each folder's own member spread "
        "sets the bar"
    ),
    assign_no_categories=(
        "No usable subfolders in {target} (need at least one subfolder with "
        "supported files)"
    ),
    assign_no_new_files="No new files to sort",
    assign_summary=(
        "{assigned} files match an existing folder; {unsorted} are not close "
        "enough and stay in place"
    ),
    assign_decision="{file} -> {folder} (similarity {similarity:.3f}, bar {floor:.3f})",
    assign_decision_unsorted=(
        "{file} stays (closest: {folder}, similarity {similarity:.3f}, bar "
        "{floor})"
    ),
    organize_cancelled="Cancelled",
    organize_dry_run_hint=(
        "This is preview mode. Use --execute to run or --output-script to "
        "generate script."
    ),
    organize_script_header="DITE File Organization Script",
    organize_script_usage="Usage:",
    organize_script_create_dirs="Create directories",
    organize_script_copy_files="Copy files (using cp + rm instead of mv for safety)",
    organize_script_exists="Stopped, so as not to overwrite an existing file:",
    organize_script_verify="Delete original files after verification",
    organize_script_done="✅ Done! Copied {count} files",
    organize_script_warning=(
        "⚠️  Original files not deleted, please confirm and delete manually"
    ),
    # Cache commands
    cache_cleared="Cleared {count} cache entries",
    cache_vlm_cleared=(
        "Cleared {count} VLM cache entries and invalidated {embeddings} embeddings "
        "built on them (document conversion results and other embeddings preserved)"
    ),
    cache_vacuum_failed=(
        "Could not compact the cache file ({error}). Space from evicted entries "
        "will be reused, but the file stays above cache.max_size_gb until a later "
        "compaction succeeds. Free some disk space or run `dite cache clear`."
    ),
    cache_status_title="Cache Status",
    cache_db_path="Database path:",
    cache_total_entries="Total entries:",
    cache_with_embedding="With embedding:",
    cache_current_embedding="Current embedding:",
    cache_stale_embedding="Stale embedding:",
    cache_embedding_version="Current embedding cache version:",
    cache_with_vlm="With VLM fallback:",
    cache_vlm_version="VLM cache version:",
    cache_unique_hashes="Unique hashes:",
    cache_db_size="Database size:",
    # Errors
    error_copy_failed="Copy verification failed",
    error_api_connection_failed=(
        "API connection failed. Please check api.base_url, api.api_key, network, "
        "and proxy settings."
    ),
    error_api_request_failed="API request failed: {error}",
    error_processing_failed="Processing failed: {error}",
    error_interrupted="Interrupted; the run was stopped.",
    error_docling_pdf_models_missing=(
        "Docling PDF models are not installed. Run `{command}` first."
    ),
    error_docling_pdf_timeout="Docling PDF extraction timed out after {seconds:g}s",
    error_docling_pdf_subprocess_failed=(
        "Docling PDF subprocess exited without a result (exit code {exitcode})"
    ),
    error_pdf_render_failed="PDF rendering failed",
    error_pdf_vlm_no_usable_content="VLM fallback returned no usable content",
    error_text_decode_failed=(
        "Unable to decode file with attempted encodings: {encodings}"
    ),
    error_vlm_client_not_initialized="VLM client is not initialized",
    # Progress
    progress_executing="Executing",
    progress_document_conversion="Executing document conversion...",
    progress_vlm_fallback="VLM fallback needed (insufficient content)",
    progress_vlm_api_call="Calling VLM API...",
    # CLI help texts
    cli_help_verbose="Verbose output",
    cli_help_quiet="Quiet mode",
    cli_help_color="Force color output",
    cli_help_version="Show version",
    cli_help_folder_scan="Folder to scan",
    cli_help_folder_pdf_check="Folder with PDF files to check",
    cli_help_output_report="Output JSON report path",
    cli_help_disable_cache="Disable cache",
    cli_help_cached_vlm_only="Use cached VLM results only; do not call the VLM API",
    cli_help_disable_knn_repair="Disable noise repair",
    cli_help_folder_organize="Folder to organize",
    cli_help_target_folder="Target folder",
    cli_help_preview_mode="Preview mode",
    cli_help_execute_move="Execute file move",
    cli_help_output_script="Output shell script path",
    cli_help_cache_group="Manage cache",
    cli_help_cache_clear="Clear all cache",
    cli_help_cache_clear_vlm="Clear VLM cache only",
    cli_help_cache_status="View cache status",
    cli_help_setup_group="Set up local capabilities",
    cli_help_setup_docling_pdf="Install the local Docling PDF models required by DITE",
    cli_help_setup_docling_pdf_force="Force re-download of models",
    cli_help_setup_docling_pdf_progress="Show download progress",
    # Setup command
    setup_docling_pdf_start="Installing Docling PDF models to: {path}",
    setup_docling_pdf_done="Docling PDF models are ready: {path}",
    setup_docling_pdf_failed="Failed to install Docling PDF models: {error}",
    setup_docling_pdf_incomplete="Docling PDF models are incomplete: {path}",
    # Logging
    log_debug_prefix="DEBUG:",
    log_info_prefix="INFO:",
    log_warning_prefix="WARNING:",
    log_error_prefix="ERROR:",
    log_success_prefix="SUCCESS:",
    # Debug messages
    debug_duplicate_groups="Duplicate file groups detected:",
    debug_duplicate_group_hash="  [hash]{hash}...[/hash]",
    debug_duplicate_group_file="    - [path]{name}[/path]",
    debug_scan_folder="Scan folder: [path]{folder}[/path]",
    debug_scan_recursive="Recursive scan: {recursive}",
    debug_scan_extensions="Supported extensions: {extensions}",
    debug_scan_excluded_dirs="Excluded directories: {paths}",
    debug_scan_no_extension_label="(no extension)",
    debug_scan_skipped_unsupported="Skipped {count} unsupported files",
    debug_scan_skipped_extension_count="  {extension}: {count}",
    debug_scan_skipped_ignored=(
        "Skipped {dirs} ignored directories and {files} ignored files"
    ),
    debug_scan_summary=(
        "Scan completed: {supported} supported, {excluded} excluded by path, "
        "{skipped} unsupported"
    ),
    debug_extract_processing_file="Processing file: [path]{path}[/path]",
    debug_extract_hash="  File hash: [hash]{hash}[/hash]",
    debug_extract_doc_cache_hit="  Doc cache hit",
    debug_extract_doc_cache_duplicate_source=(
        "    Reused cached source: [path]{source}[/path]"
    ),
    debug_extract_doc_cache_miss="  Doc cache miss",
    debug_extract_doc_result=(
        "  Document extraction: extractor={extractor}, success={success}, "
        "length={length}, error={error}"
    ),
    debug_extract_vlm_check=(
        "  VLM check: suffix={suffix}, effective_length={effective_length}, "
        "threshold={threshold}, needed={needed}"
    ),
    debug_pdf_profile=(
        "  PDF profile: kind={kind}, reason={reason}, "
        "effective_length={effective_length}, glyph_noise={glyph_noise_tokens}, "
        "fallback={needs_vlm_fallback}"
    ),
    debug_extract_vlm_cache_hit="  VLM cache hit (length={length})",
    debug_extract_vlm_api_call="  Calling VLM fallback",
    debug_extract_vlm_result=(
        "  VLM result: success={success}, length={length}, error={error}"
    ),
    debug_extract_vlm_selected="  Using VLM result ({vlm_length} > {doc_length})",
    debug_extract_vlm_skipped=(
        "  Keeping document extraction ({doc_length} >= {vlm_length})"
    ),
    debug_extract_truncated="  Content truncated: {original} -> {limit}",
    debug_extract_summary=(
        "Extraction summary: doc_cache_hits={doc_cache_hits}, "
        "vlm_cache_hits={vlm_cache_hits}, primary_failures={primary_failures}, "
        "fallback_needed={source_fallback_needed}, "
        "selected_vlm_files={selected_vlm_files}, "
        "vlm_api_page_calls={vlm_api_page_calls}, duplicates={duplicates}"
    ),
    debug_vector_cache_summary="Embedding cache: {hits} hits, {misses} misses",
    debug_vectorizing_documents="Vectorizing {count} documents",
    debug_vectorizing_model="Model: {model}",
    debug_vector_fallback_names="  {count} files fell back to file names: {names}",
    debug_vector_text_stats=(
        "  Text length: min={min_length}, max={max_length}, avg={avg_length}"
    ),
    debug_vector_dimension="  Vector dimension: {dimension}",
    debug_vector_api_usage="  API usage: {tokens} tokens",
    debug_vector_batch_split=(
        "  Embedding batch rejected (status={status}); "
        "splitting {size} inputs in half"
    ),
    debug_cluster_hdbscan_header="HDBSCAN parameters:",
    debug_cluster_hdbscan_min_cluster_size="  min_cluster_size: {value}",
    debug_cluster_hdbscan_min_samples="  min_samples: {value}",
    debug_cluster_hdbscan_epsilon="  cluster_selection_epsilon: {value}",
    debug_cluster_hdbscan_method="  cluster_selection_method: {value}",
    debug_cluster_input_vectors="  Input vectors: {count}, dimension {dimension}",
    debug_cluster_initial_result=(
        "Initial clustering: {clusters} clusters, {noise} noise points"
    ),
    debug_cluster_sizes="  Cluster sizes: {sizes}",
    debug_cluster_knn_dynamic_threshold=(
        "k-NN dynamic threshold: {threshold:.4f} "
        "(mean_core_distance={mean_core_distance:.4f})"
    ),
    debug_cluster_knn_fixed_threshold="k-NN fixed threshold: {threshold:.4f}",
    debug_cluster_knn_assignment=(
        "  Noise point {index} ({name}) -> cluster {label} "
        "(distance={distance:.4f}, threshold={threshold:.4f})"
    ),
    debug_cluster_knn_kept=(
        "  Noise point {index} ({name}) kept as noise, candidate cluster {label} "
        "(distance={distance:.4f}, threshold={threshold:.4f})"
    ),
    debug_cluster_knn_summary=(
        "k-NN repaired {repaired} noise points and kept {kept} as noise"
    ),
    debug_cluster_small_merge_event=(
        "  Small cluster {source} (size={source_size}) -> cluster {target} "
        "(target_size={target_size}, similarity={similarity:.4f})"
    ),
    debug_cluster_small_merge_skipped=(
        "  Small cluster {source} (size={source_size}) skipped, best target {target} "
        "(target_size={target_size}, similarity={similarity:.4f}, reason={reason})"
    ),
    debug_cluster_small_merge_summary=(
        "Small-cluster merge: candidates={candidates}, merged={merged}, "
        "skipped={skipped}"
    ),
    debug_cluster_name_empty_response=(
        "Cluster naming returned empty content: model={model}, "
        "finish_reason={finish_reason}, reasoning_chars={reasoning_chars}"
    ),
    debug_cluster_name_retry=(
        "Cluster naming request failed (attempt {attempt}/{max_retries}): {error}"
    ),
    debug_cluster_name_empty_fallback=(
        "Cluster naming returned empty content, using fallback: {fallback}"
    ),
    debug_cluster_name_invalid_fallback=(
        "Cluster naming returned invalid name, using fallback: {fallback}"
    ),
    debug_cluster_name_cut_off_fallback=(
        "Cluster naming reply was cut off by max_tokens before its first line "
        "ended, using fallback: {fallback}"
    ),
    debug_cluster_name_failed_fallback=(
        "Cluster naming failed, using fallback {fallback}: {error}"
    ),
    debug_cluster_merge="Merged cluster {source} -> {target} (name: {name})",
    debug_cluster_name_result="Cluster {label} named {name} (files={count})",
    debug_cluster_name_async_request_failed=(
        "Cluster {label} async naming request failed: error={error}, "
        "wait={wait_sec:.3f}s, request={request_sec:.3f}s, fallback={fallback}"
    ),
    debug_cluster_name_async_response_empty=(
        "Cluster {label} async naming response empty: "
        "wait={wait_sec:.3f}s, request={request_sec:.3f}s, fallback={fallback}"
    ),
    debug_cluster_name_async_response_invalid=(
        "Cluster {label} async naming response invalid: "
        "wait={wait_sec:.3f}s, request={request_sec:.3f}s, fallback={fallback}"
    ),
    warning_cluster_naming_fallback=(
        "LLM cluster naming fell back to heuristic names for {fallback} of "
        "{total} clusters (request failed: {failed}, empty, reasoning-only or "
        "cut-off reply: {empty}, unusable name: {invalid}). Run with --verbose "
        "for per-cluster details."
    ),
    warning_cluster_naming_first_error="First cluster naming error: {error}",
    warning_cluster_naming_reasoning_hint=(
        "Reasoning may be using up the naming output: keep "
        "request_profiles.cluster_naming.reasoning_mode: off, set "
        "api.reasoning_param_style for your provider (see docs/configuration.md), "
        "or raise its max_tokens."
    ),
    debug_analyzer_json_parse_failed=(
        "JSON parsing failed (attempt {attempt}/{max_retries}): {error}"
    ),
    debug_analyzer_api_failed=(
        "API call failed (attempt {attempt}/{max_retries}): {error}"
    ),
    warning_unsupported_file_format="Unsupported file format: {suffix}",
    warning_extractor_failed="{extractor} failed on {name}: {error}",
    warning_extract_file_failed=(
        "Extraction failed for {path}; continuing with the other files: {error}"
    ),
    warning_extract_cache_failed=(
        "Cache read/write failed for {path}; its extracted content is kept: {error}"
    ),
    warning_pdf2image_missing="pdf2image is not installed, cannot use VLM fallback",
    warning_vlm_fallback_failed="VLM fallback failed: {error}",
    warning_analyzer_default_used="Document analysis failed, using defaults",
    warning_analyzer_failed="Document analysis failed: {error}",
    debug_vlm_page_processing="  Processing page {page}/{total} ({width}x{height})",
    debug_vlm_page_resized=(
        "  Resized: {old_width}x{old_height} -> {new_width}x{new_height}"
    ),
    debug_vlm_image_size="  Image size: {size_kb:.1f} KB",
    debug_vlm_api_call="  Calling VLM API...",
    debug_vlm_page_done="  Page {page} completed ({length} chars)",
    debug_vlm_page_failed="  Page {page} failed: {error}",
    debug_vlm_page_result_done=(
        "  Page {page} result: success, length={length}, "
        "wait={wait_sec:.3f}s, request={request_sec:.3f}s"
    ),
    debug_vlm_page_result_failed=(
        "  Page {page} result: failed, error={error}, "
        "wait={wait_sec:.3f}s, request={request_sec:.3f}s"
    ),
    debug_vlm_page_result_empty=(
        "  Page {page} result: empty, wait={wait_sec:.3f}s, "
        "request={request_sec:.3f}s"
    ),
    debug_vlm_batch_summary=(
        "  VLM batch summary: success={success}, failed={failed}, "
        "max_wait={max_wait_sec:.3f}s, max_request={max_request_sec:.3f}s"
    ),
    debug_async_runtime_limits=(
        "Async API runtime limits: cluster={cluster_workers}, vlm={vlm_workers}, "
        "vlm_pages_per_document={vlm_pages_per_document}, "
        "max_connections={max_connections}"
    ),
)

# Locale to Messages mapping
_TRANSLATIONS: dict[Locale, Messages] = {
    "zh-CN": ZH_CN,
    "en": EN,
}

# Current locale
_current_locale: Locale = DEFAULT_LOCALE
_current_messages: Messages = _TRANSLATIONS[DEFAULT_LOCALE]


def _normalize_locale(locale: str) -> Locale:
    """Normalize locale aliases to supported locale keys."""
    lowered = locale.strip().lower().replace("_", "-")

    if lowered in {"en", "en-us", "en-gb"}:
        return "en"

    if lowered in {"zh", "zh-cn", "zh-hans"}:
        return "zh-CN"

    raise ValueError(
        f"Unsupported locale: {locale}. Supported: {list(_TRANSLATIONS.keys())}"
    )


def set_locale(locale: Locale | str) -> None:
    """Set the current locale."""
    global _current_locale, _current_messages
    normalized = _normalize_locale(locale)
    _current_locale = normalized
    _current_messages = _TRANSLATIONS[normalized]


def get_locale() -> Locale:
    """Get the current locale."""
    return _current_locale


def get_messages() -> Messages:
    """Get the current messages object."""
    return _current_messages


def t(key: str, **kwargs) -> str:
    """
    Get a translated message by key with optional interpolation.

    Args:
        key: Message key (attribute name in Messages)
        **kwargs: Values for string interpolation

    Returns:
        Translated and interpolated string
    """
    msg = getattr(_current_messages, key, key)
    if kwargs:
        msg = msg.format(**kwargs)

    # Remove emoji characters for consistent output
    for token in ("📁", "✅", "⚠️", "✓"):
        msg = msg.replace(token, "")

    return msg.strip()
