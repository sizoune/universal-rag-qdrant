"""
Tree-sitter based semantic code parser.
Extracts functions, classes, and methods as individual chunks
instead of arbitrary character-based splitting.
"""

import os
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Language configs: extension → (grammar module, top-level node types to extract)
LANGUAGE_MAP = {
    ".py": {
        "module": "tree_sitter_python",
        "node_types": [
            "function_definition",
            "class_definition",
            "decorated_definition",
        ],
    },
    ".js": {
        "module": "tree_sitter_javascript",
        "node_types": [
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
        ],
    },
}

# Max chunk size — if a single block exceeds this, sub-split it
MAX_BLOCK_SIZE = 1500


def _get_parser(ext: str):
    """Create a Tree-sitter parser for the given file extension.
    Returns (parser, language, node_types) or None if unsupported."""
    lang_config = LANGUAGE_MAP.get(ext)
    if not lang_config:
        return None

    try:
        from tree_sitter import Language, Parser
        import importlib

        grammar_module = importlib.import_module(lang_config["module"])
        language = Language(grammar_module.language())
        parser = Parser(language)
        return parser, language, lang_config["node_types"]
    except (ImportError, Exception) as e:
        logger.warning(f"Tree-sitter not available for '{ext}': {e}")
        return None


def _extract_node_name(node, source_bytes: bytes) -> str:
    """Try to extract a meaningful name from an AST node."""
    # For decorated definitions, dig into the actual definition
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                name_node = child.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode("utf-8")

    # For export_statement, dig into the declaration
    if node.type == "export_statement":
        for child in node.children:
            if child.type in (
                "function_declaration",
                "class_declaration",
                "lexical_declaration",
            ):
                name_node = child.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode("utf-8")

    # Direct name field
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8")

    return f"<{node.type}>"


def parse_code_file(filepath: str) -> list[Document]:
    """Parse a code file using Tree-sitter into semantic chunks.

    Each function/class becomes its own Document. Falls back to
    RecursiveCharacterTextSplitter if Tree-sitter is unavailable
    or for blocks exceeding MAX_BLOCK_SIZE.

    Returns list of Documents with metadata including language and node info.
    """
    ext = os.path.splitext(filepath)[1].lower()
    abs_path = os.path.abspath(filepath)

    # Try to get Tree-sitter parser
    result = _get_parser(ext)
    if result is None:
        logger.info(f"No Tree-sitter support for '{ext}', using text fallback")
        return None  # Signal to caller to use default loader

    parser, language, target_node_types = result

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source_code = f.read()
    except (IOError, UnicodeDecodeError) as e:
        logger.error(f"Failed to read '{filepath}': {e}")
        return None

    source_bytes = source_code.encode("utf-8")
    tree = parser.parse(source_bytes)
    root_node = tree.root_node

    documents = []
    covered_ranges = []  # Track which byte ranges are covered
    lang_name = ext.lstrip(".")

    # Extract top-level semantic blocks
    for child in root_node.children:
        if child.type in target_node_types:
            block_text = source_bytes[child.start_byte : child.end_byte].decode("utf-8")
            node_name = _extract_node_name(child, source_bytes)

            if len(block_text) > MAX_BLOCK_SIZE:
                # Sub-split large blocks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=100
                )
                sub_chunks = splitter.split_text(block_text)
                for i, chunk in enumerate(sub_chunks):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": abs_path,
                                "source_type": "local",
                                "language": lang_name,
                                "node_type": child.type,
                                "node_name": f"{node_name} (part {i+1})",
                                "parser": "tree-sitter",
                            },
                        )
                    )
            else:
                documents.append(
                    Document(
                        page_content=block_text,
                        metadata={
                            "source": abs_path,
                            "source_type": "local",
                            "language": lang_name,
                            "node_type": child.type,
                            "node_name": node_name,
                            "parser": "tree-sitter",
                        },
                    )
                )

            covered_ranges.append((child.start_byte, child.end_byte))

    # Collect uncovered top-level code (imports, constants, etc.)
    uncovered_text = []
    last_end = 0
    for start, end in sorted(covered_ranges):
        gap = source_bytes[last_end:start].decode("utf-8").strip()
        if gap:
            uncovered_text.append(gap)
        last_end = end

    # Trailing code
    trailing = source_bytes[last_end:].decode("utf-8").strip()
    if trailing:
        uncovered_text.append(trailing)

    if uncovered_text:
        combined = "\n\n".join(uncovered_text)
        if len(combined) > MAX_BLOCK_SIZE:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            for chunk in splitter.split_text(combined):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": abs_path,
                            "source_type": "local",
                            "language": lang_name,
                            "node_type": "module_scope",
                            "node_name": "<imports/constants>",
                            "parser": "tree-sitter",
                        },
                    )
                )
        else:
            documents.append(
                Document(
                    page_content=combined,
                    metadata={
                        "source": abs_path,
                        "source_type": "local",
                        "language": lang_name,
                        "node_type": "module_scope",
                        "node_name": "<imports/constants>",
                        "parser": "tree-sitter",
                    },
                )
            )

    logger.info(
        f"Tree-sitter parsed '{filepath}': {len(documents)} semantic chunks "
        f"({sum(1 for d in documents if d.metadata.get('node_type') != 'module_scope')} "
        f"functions/classes)"
    )
    return documents
