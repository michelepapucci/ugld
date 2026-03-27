import re

with open("site/ugld.html") as f:
    html = f.read()

# 1. inject the back link into the nav
html = html.replace(
    "<div>\n\n\n\n\n            <h2>API Documentation</h2>",
    '<div>\n            <a href="index.html" class="back-to-index">&#8592; Back to index</a>\n\n\n\n\n            <h2>API Documentation</h2>',
    1,
)

# 2. inject styles into the existing custom.css block
css = """
.back-to-index {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    color: var(--text);
    text-decoration: none;
    font-size: 0.85rem;
    opacity: 0.65;
    transition: opacity 150ms;
    margin-bottom: 1rem;
    padding-left: clamp(0.5rem, 2vw, 1.75rem);
}
.back-to-index:hover { opacity: 1; }
"""
html = html.replace("/*! custom.css */", "/*! custom.css */" + css, 1)

with open("site/ugld.html", "w") as f:
    f.write(html)
