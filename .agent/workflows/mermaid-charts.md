---
description: How to create mermaid charts that render reliably in VS Code markdown preview
---

# Mermaid Chart Best Practices

Follow these rules to prevent "No diagram type detected" errors.

## Rules

1. **Use `graph` not `flowchart`** - `flowchart` may fail in older mermaid renderers
2. **No HTML tags** in node labels - avoid `<br/>`, use colons/commas instead
3. **No Unicode** - use `phi_1` not `φ₁`, `theta` not `θ`
4. **Quote labels** with special characters like `>`, `<`, or operators: `{"p > 0.05?"}`
5. **No UTF-8 BOM** - files with BOM (`EF BB BF`) may cause mermaid parse failures; save as UTF-8 without BOM

## Quick Reference

| Issue | Bad | Good |
|-------|-----|------|
| Diagram type | `flowchart LR` | `graph LR` |
| Line breaks | `"Line1<br/>Line2"` | `"Line1: Line2"` |
| Greek letters | `"φ₁"` | `"phi_1"` |
| Special chars | `{p > 0.05?}` | `{"p > 0.05?"}` |

## Troubleshooting

If "No diagram type detected":
1. Ensure `graph` not `flowchart`
2. Remove `<br/>` tags
3. Replace Unicode with ASCII
4. Quote special-char labels
5. Remove UTF-8 BOM from file
