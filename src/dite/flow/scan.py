"""文件扫描模块"""

import os
from pathlib import Path

from dite.app.config import Config
from dite.app.i18n import t
from dite.util.log import get_logger

# 扫描时跳过的条目。这些规则只检查扫描根目录之下的目录名和文件名：用户直接指定
# 的根目录本身即使是隐藏目录、应用包或虚拟环境，也照常扫描它的内容。被跳过的
# 目录在遍历时就地剪掉，不会再往下列目录。名称比较不区分大小写（macOS 和
# Windows 的默认文件系统都不区分）。

# 以 "." 开头的目录和文件都视为隐藏条目。这一条同时覆盖了 .git、.venv、.obsidian、
# .Trash、dite 自己写移动记录的 .dite 目录，以及 .DS_Store、macOS 在非 HFS 卷和
# 压缩包里生成的 AppleDouble 元数据文件 "._*"、LibreOffice 的锁文件 ".~lock.*#"。
HIDDEN_NAME_PREFIX = "."

# 依赖、缓存和系统目录：里面是工具安装或生成的文件，不是用户的文档。
SKIPPED_DIR_NAMES = frozenset(
    {
        "node_modules",  # npm / yarn / pnpm 安装的依赖
        "__pycache__",  # Python 字节码缓存
        "site-packages",  # Python 包安装目录（venv、conda、系统 Python）
        "__macosx",  # macOS 压缩包解开后的 AppleDouble 元数据目录
        "$recycle.bin",  # Windows 回收站
        "system volume information",  # Windows 卷的系统目录
    }
)

# 目录里有这个文件，说明它是 Python 虚拟环境（venv 和 virtualenv 都会写入）；
# 虚拟环境的目录名不一定以点开头，例如 venv、env。
VIRTUALENV_MARKER = "pyvenv.cfg"

# macOS 包目录的后缀：Finder 把这类目录当成单个文件显示，里面的文件属于应用、
# 图库或文档本身，被整理搬走会损坏它们。只对目录生效，同名后缀的普通文件不受影响。
PACKAGE_DIR_SUFFIXES = frozenset(
    {
        # 应用、插件和系统扩展
        ".app",
        ".appex",
        ".bundle",
        ".component",
        ".framework",
        ".kext",
        ".mdimporter",
        ".plugin",
        ".prefpane",
        ".qlgenerator",
        ".saver",
        ".vst",
        ".vst3",
        ".xpc",
        # 安装包
        ".pkg",
        ".mpkg",
        # 照片、音乐、视频图库和音视频工程
        ".photoslibrary",
        ".photolibrary",
        ".aplibrary",
        ".lrlibrary",
        ".lrdata",
        ".musiclibrary",
        ".tvlibrary",
        ".imovielibrary",
        ".theater",
        ".fcpbundle",
        ".logicx",
        ".band",
        # 开发工具的工程和产物
        ".xcodeproj",
        ".xcworkspace",
        ".xcarchive",
        ".playground",
        ".dsym",
        ".docset",
        # 以包形式保存的文档
        ".rtfd",
        ".pages",
        ".numbers",
        ".key",
        ".scptd",
        # 磁盘映像和虚拟机
        ".sparsebundle",
        ".vmwarevm",
        ".pvm",
        ".utm",
    }
)

# 临时文件和系统文件：Microsoft Office 打开文档时生成的锁文件 "~$*"，以及
# Windows 资源管理器生成的 Thumbs.db、desktop.ini（.DS_Store、"._*" 已被隐藏规则覆盖）。
SKIPPED_FILE_PREFIXES = ("~$",)
SKIPPED_FILE_NAMES = frozenset({"thumbs.db", "desktop.ini"})


def _is_skipped_dir(parent: Path, name: str) -> bool:
    """根目录之下的子目录 ``parent / name`` 是否应整个跳过。"""
    lowered = name.lower()
    return (
        name.startswith(HIDDEN_NAME_PREFIX)
        or lowered in SKIPPED_DIR_NAMES
        or os.path.splitext(lowered)[1] in PACKAGE_DIR_SUFFIXES
        # 用 os.path.isfile 而不是 Path.is_file：后者遇到无权进入的目录（例如
        # 挂载盘上 root 的 lost+found、macOS 受隐私保护的目录）会抛出
        # PermissionError，让整次扫描中止；前者吞掉 OSError 返回 False，
        # 随后 os.walk 列不出这个目录，就把它跳过。
        or os.path.isfile(parent / name / VIRTUALENV_MARKER)
    )


def _is_skipped_file(name: str) -> bool:
    """名为 ``name`` 的文件是否是隐藏、临时或系统文件。"""
    return (
        name.startswith(HIDDEN_NAME_PREFIX)
        or name.startswith(SKIPPED_FILE_PREFIXES)
        or name.lower() in SKIPPED_FILE_NAMES
    )


def scan_files(
    folder: Path,
    *,
    config: Config,
    extensions: set[str] | None = None,
    recursive: bool = True,
    exclude_paths: list[Path] | None = None,
) -> list[Path]:
    """
    扫描文件夹，返回支持的文件列表

    隐藏条目、依赖目录、虚拟环境、macOS 包目录以及临时和系统文件会被跳过，
    规则见本模块顶部的常量。

    Args:
        folder: 要扫描的文件夹路径
        extensions: 要扫描的文件扩展名集合，默认使用配置
        recursive: 是否递归扫描子目录
        exclude_paths: 需要排除的目录路径列表

    Returns:
        按路径排序的文件列表
    """
    logger = get_logger()

    if extensions is None:
        extensions = config.formats.all_extensions

    logger.debug(t("debug_scan_folder", folder=folder))
    logger.debug(t("debug_scan_recursive", recursive=recursive))
    logger.debug(t("debug_scan_extensions", extensions=", ".join(sorted(extensions))))

    excludes = [p.resolve() for p in (exclude_paths or [])]
    if excludes:
        logger.debug(
            t(
                "debug_scan_excluded_dirs",
                paths=", ".join(f"[path]{str(path)}[/path]" for path in excludes),
            )
        )

    files = []
    excluded_count = 0
    skipped_count = 0
    skipped_extensions: dict[str, int] = {}
    ignored_dir_count = 0
    ignored_file_count = 0

    # os.walk 自顶向下遍历，就地修改 dirnames 剪掉要跳过的目录，避免把
    # node_modules 这类巨大目录整棵列一遍再过滤。不跟随目录符号链接。
    for dirpath, dirnames, filenames in os.walk(folder):
        directory = Path(dirpath)
        if recursive:
            kept = [name for name in dirnames if not _is_skipped_dir(directory, name)]
            ignored_dir_count += len(dirnames) - len(kept)
            dirnames[:] = kept
        else:
            dirnames.clear()

        for name in filenames:
            if _is_skipped_file(name):
                ignored_file_count += 1
                continue

            file = directory / name
            if not file.is_file():
                continue

            resolved_file = file.resolve()
            if any(resolved_file.is_relative_to(exclude) for exclude in excludes):
                excluded_count += 1
                continue

            if file.suffix.lower() in extensions:
                files.append(file)
            else:
                skipped_count += 1
                ext = (
                    file.suffix.lower()
                    if file.suffix
                    else t("debug_scan_no_extension_label")
                )
                skipped_extensions[ext] = skipped_extensions.get(ext, 0) + 1

    if ignored_dir_count or ignored_file_count:
        logger.debug(
            t(
                "debug_scan_skipped_ignored",
                dirs=ignored_dir_count,
                files=ignored_file_count,
            )
        )

    if skipped_count > 0:
        logger.debug(t("debug_scan_skipped_unsupported", count=skipped_count))
        if skipped_extensions:
            top_skipped = sorted(
                skipped_extensions.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for ext, count in top_skipped:
                logger.debug(
                    t(
                        "debug_scan_skipped_extension_count",
                        extension=ext,
                        count=count,
                    )
                )

    logger.debug(
        t(
            "debug_scan_summary",
            supported=len(files),
            excluded=excluded_count,
            skipped=skipped_count,
        )
    )

    return sorted(files)
