#!/usr/bin/env python3
"""
Documentation Generator for Defect Detection System
Automatically generates documentation for the codebase.
"""

import os
import sys
import glob
import re
import shutil
import argparse
import importlib.util
import inspect
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Try to import Markdown module
try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

# Custom function to safely import a module from file
def import_module_from_file(module_path):
    """
    Import a module from a file path.
    
    Args:
        module_path: Path to the Python module
        
    Returns:
        The imported module or None if import fails
    """
    try:
        module_name = os.path.basename(module_path).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            return None
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing {module_path}: {e}")
        return None

class DocGenerator:
    """
    Generator for code documentation.
    """
    
    def __init__(self, source_dir, output_dir, project_name="Defect Detection System"):
        """
        Initialize the documentation generator.
        
        Args:
            source_dir: Directory containing source code
            output_dir: Directory to write documentation
            project_name: Name of the project
        """
        self.source_dir = os.path.abspath(source_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.project_name = project_name
        self.nav_links = []
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_documentation(self):
        """
        Generate documentation for all Python files in the source directory.
        """
        print(f"Generating documentation from {self.source_dir} to {self.output_dir}")
        
        # Create documentation structure
        self._create_structure()
        
        # Generate documentation for each Python file
        python_files = self._get_python_files()
        modules_doc = []
        
        for file_path in python_files:
            try:
                module_doc = self._document_file(file_path)
                if module_doc:
                    modules_doc.append(module_doc)
            except Exception as e:
                print(f"Error documenting {file_path}: {e}")
        
        # Generate index page
        self._generate_index(modules_doc)
        
        # Generate CSS file
        self._generate_css()
        
        print(f"Documentation generated successfully in {self.output_dir}")
    
    def _get_python_files(self):
        """
        Get all Python files in the source directory.
        
        Returns:
            List of Python file paths
        """
        python_files = []
        
        # Walk through source directory
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        
        return sorted(python_files)
    
    def _create_structure(self):
        """Create the documentation directory structure"""
        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        modules_dir = os.path.join(self.output_dir, "modules")
        os.makedirs(modules_dir, exist_ok=True)
        
        css_dir = os.path.join(self.output_dir, "css")
        os.makedirs(css_dir, exist_ok=True)
    
    def _document_file(self, file_path):
        """
        Generate documentation for a single Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary with module documentation info
        """
        rel_path = os.path.relpath(file_path, self.source_dir)
        print(f"Documenting {rel_path}")
        
        # Extract module name
        module_name = os.path.basename(file_path).replace(".py", "")
        
        # Import the module
        module = import_module_from_file(file_path)
        
        if module is None:
            print(f"Skipping {file_path} due to import error")
            return None
        
        # Extract module docstring
        module_doc = inspect.getdoc(module) or "No module documentation"
        
        # Get all classes and functions
        classes = []
        functions = []
        
        for name, obj in inspect.getmembers(module):
            # Skip private members and imported objects
            if name.startswith("_") or inspect.getmodule(obj) != module:
                continue
            
            if inspect.isclass(obj):
                class_info = self._document_class(obj)
                classes.append(class_info)
            elif inspect.isfunction(obj):
                func_info = self._document_function(obj)
                functions.append(func_info)
        
        # Create module documentation
        module_info = {
            "name": module_name,
            "file_path": file_path,
            "rel_path": rel_path,
            "doc": module_doc,
            "classes": classes,
            "functions": functions
        }
        
        # Generate HTML file
        self._generate_module_html(module_info)
        
        return module_info
    
    def _document_class(self, cls):
        """
        Generate documentation for a class.
        
        Args:
            cls: Class object
            
        Returns:
            Dictionary with class documentation info
        """
        # Extract class docstring
        class_doc = inspect.getdoc(cls) or "No class documentation"
        
        # Get methods
        methods = []
        
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            # Skip private methods
            if name.startswith("_") and name != "__init__":
                continue
            
            method_info = self._document_function(method)
            methods.append(method_info)
        
        # Create class documentation
        class_info = {
            "name": cls.__name__,
            "doc": class_doc,
            "methods": methods
        }
        
        return class_info
    
    def _document_function(self, func):
        """
        Generate documentation for a function.
        
        Args:
            func: Function object
            
        Returns:
            Dictionary with function documentation info
        """
        # Extract function docstring
        func_doc = inspect.getdoc(func) or "No function documentation"
        
        # Get signature
        try:
            signature = str(inspect.signature(func))
        except ValueError:
            signature = "()"
        
        # Extract parameter info from docstring
        params = self._parse_docstring_params(func_doc)
        
        # Extract return info from docstring
        returns = self._parse_docstring_returns(func_doc)
        
        # Create function documentation
        func_info = {
            "name": func.__name__,
            "signature": signature,
            "doc": func_doc,
            "params": params,
            "returns": returns
        }
        
        return func_info
    
    def _parse_docstring_params(self, docstring):
        """
        Parse parameter information from docstring.
        
        Args:
            docstring: Function docstring
            
        Returns:
            List of parameter dictionaries
        """
        params = []
        
        # Regular expression to match parameter documentation
        # Matches lines like: "    param_name: param_description"
        # or "    Args:\n        param_name: param_description"
        param_pattern = r'(?:Args:|Parameters:)\s*(?:\n\s+(\w+):\s*(.*?)(?:\n\s+\w+:|$))'
        
        matches = re.findall(param_pattern, docstring, re.DOTALL)
        
        for name, description in matches:
            params.append({
                "name": name.strip(),
                "description": description.strip()
            })
        
        return params
    
    def _parse_docstring_returns(self, docstring):
        """
        Parse return information from docstring.
        
        Args:
            docstring: Function docstring
            
        Returns:
            Return description or None
        """
        # Regular expression to match return documentation
        # Matches lines like: "    Returns:\n        return_description"
        return_pattern = r'Returns:\s*(.*?)(?:\n\s*\w+:|$)'
        
        match = re.search(return_pattern, docstring, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return None
    
    def _generate_module_html(self, module_info):
        """
        Generate HTML documentation for a module.
        
        Args:
            module_info: Dictionary with module documentation info
        """
        module_name = module_info["name"]
        module_doc = module_info["doc"]
        classes = module_info["classes"]
        functions = module_info["functions"]
        
        # Create output file path
        rel_dir = os.path.dirname(module_info["rel_path"])
        output_dir = os.path.join(self.output_dir, "modules", rel_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{module_name}.html")
        
        # Add to navigation links
        nav_path = f"modules/{rel_dir}/{module_name}.html" if rel_dir else f"modules/{module_name}.html"
        self.nav_links.append({
            "name": module_name,
            "path": nav_path
        })
        
        # Create HTML content
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{module_name} - {self.project_name}</title>
    <link rel="stylesheet" href="{'../' * (len(rel_dir.split('/')) + 1)}css/style.css">
</head>
<body>
    <header>
        <h1>{self.project_name}</h1>
        <nav>
            <a href="{'../' * (len(rel_dir.split('/')) + 1)}index.html">Home</a>
        </nav>
    </header>
    
    <main>
        <h2>Module: {module_name}</h2>
        <div class="module-path">{module_info["rel_path"]}</div>
        
        <section class="module-doc">
            <div class="doc-content">
                {self._format_docstring(module_doc)}
            </div>
        </section>
        
        <section class="module-contents">
            <h3>Contents</h3>
            <ul>
"""
        
        # Add class links to contents
        for cls in classes:
            html += f'                <li><a href="#{cls["name"].lower()}">{cls["name"]}</a></li>\n'
        
        # Add function links to contents
        for func in functions:
            html += f'                <li><a href="#{func["name"].lower()}">{func["name"]}</a></li>\n'
        
        html += """            </ul>
        </section>
"""
        
        # Add classes
        if classes:
            html += """        <section class="classes">
            <h3>Classes</h3>
"""
            
            for cls in classes:
                html += self._format_class_html(cls)
            
            html += """        </section>
"""
        
        # Add functions
        if functions:
            html += """        <section class="functions">
            <h3>Functions</h3>
"""
            
            for func in functions:
                html += self._format_function_html(func)
            
            html += """        </section>
"""
        
        html += """    </main>
    
    <footer>
        <p>Generated on {date} by Documentation Generator</p>
    </footer>
</body>
</html>
""".format(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Write HTML to file
        with open(output_file, "w") as f:
            f.write(html)
    
    def _format_class_html(self, class_info):
        """
        Format class information as HTML.
        
        Args:
            class_info: Dictionary with class documentation info
            
        Returns:
            HTML string for the class
        """
        class_name = class_info["name"]
        class_doc = class_info["doc"]
        methods = class_info["methods"]
        
        html = f"""            <div class="class" id="{class_name.lower()}">
                <h4 class="class-name">{class_name}</h4>
                <div class="doc-content">
                    {self._format_docstring(class_doc)}
                </div>
"""
        
        # Add methods
        if methods:
            html += """                <div class="methods">
                    <h5>Methods</h5>
"""
            
            for method in methods:
                html += self._format_method_html(method)
            
            html += """                </div>
"""
        
        html += """            </div>
"""
        
        return html
    
    def _format_method_html(self, method_info):
        """
        Format method information as HTML.
        
        Args:
            method_info: Dictionary with method documentation info
            
        Returns:
            HTML string for the method
        """
        method_name = method_info["name"]
        method_signature = method_info["signature"]
        method_doc = method_info["doc"]
        params = method_info["params"]
        returns = method_info["returns"]
        
        html = f"""                    <div class="method" id="{method_name.lower()}">
                        <h6 class="method-name">{method_name}{method_signature}</h6>
                        <div class="doc-content">
                            {self._format_docstring(method_doc)}
                        </div>
"""
        
        # Add parameters
        if params:
            html += """                        <div class="parameters">
                            <h6>Parameters:</h6>
                            <ul>
"""
            
            for param in params:
                html += f"""                                <li><code>{param["name"]}</code>: {param["description"]}</li>
"""
            
            html += """                            </ul>
                        </div>
"""
        
        # Add return value
        if returns:
            html += f"""                        <div class="returns">
                            <h6>Returns:</h6>
                            <p>{returns}</p>
                        </div>
"""
        
        html += """                    </div>
"""
        
        return html
    
    def _format_function_html(self, func_info):
        """
        Format function information as HTML.
        
        Args:
            func_info: Dictionary with function documentation info
            
        Returns:
            HTML string for the function
        """
        func_name = func_info["name"]
        func_signature = func_info["signature"]
        func_doc = func_info["doc"]
        params = func_info["params"]
        returns = func_info["returns"]
        
        html = f"""            <div class="function" id="{func_name.lower()}">
                <h4 class="function-name">{func_name}{func_signature}</h4>
                <div class="doc-content">
                    {self._format_docstring(func_doc)}
                </div>
"""
        
        # Add parameters
        if params:
            html += """                <div class="parameters">
                    <h5>Parameters:</h5>
                    <ul>
"""
            
            for param in params:
                html += f"""                        <li><code>{param["name"]}</code>: {param["description"]}</li>
"""
            
            html += """                    </ul>
                </div>
"""
        
        # Add return value
        if returns:
            html += f"""                <div class="returns">
                    <h5>Returns:</h5>
                    <p>{returns}</p>
                </div>
"""
        
        html += """            </div>
"""
        
        return html
    
    def _format_docstring(self, docstring):
        """
        Format a docstring as HTML.
        
        Args:
            docstring: Docstring to format
            
        Returns:
            HTML representation of the docstring
        """
        if not docstring:
            return "<p>No documentation available</p>"
        
        # Clean up docstring
        docstring = docstring.strip()
        
        # Convert to HTML if markdown is available
        if HAS_MARKDOWN:
            html = markdown.markdown(docstring)
            return html
        
        # Simple conversion
        paragraphs = []
        current_paragraph = []
        
        for line in docstring.split("\n"):
            if line.strip():
                current_paragraph.append(line)
            elif current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
        
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))
        
        html = "".join([f"<p>{p}</p>" for p in paragraphs])
        return html
    
    def _generate_index(self, modules_doc):
        """
        Generate index page.
        
        Args:
            modules_doc: List of module documentation dictionaries
        """
        # Create HTML content
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.project_name} - Documentation</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <header>
        <h1>{self.project_name}</h1>
        <nav>
            <a href="index.html">Home</a>
        </nav>
    </header>
    
    <main>
        <h2>Documentation</h2>
        
        <section class="module-list">
            <h3>Modules</h3>
            <ul>
"""
        
        # Add module links
        for nav_link in self.nav_links:
            html += f'                <li><a href="{nav_link["path"]}">{nav_link["name"]}</a></li>\n'
        
        html += """            </ul>
        </section>
        
        <section class="project-overview">
            <h3>Project Overview</h3>
            <p>This documentation provides detailed information about the {project_name}.</p>
            <p>Navigate through the modules listed above to explore the codebase.</p>
        </section>
    </main>
    
    <footer>
        <p>Generated on {date} by Documentation Generator</p>
    </footer>
</body>
</html>
""".format(project_name=self.project_name, date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Write HTML to file
        with open(os.path.join(self.output_dir, "index.html"), "w") as f:
            f.write(html)
    
    def _generate_css(self):
        """Generate CSS for documentation"""
        css = """/* Documentation Styles */

/* General styles */
:root {
    --primary-color: #2c3e50;
    --secondary-color: #3498db;
    --bg-color: #f9f9f9;
    --text-color: #333;
    --code-bg: #f5f5f5;
    --border-color: #ddd;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: var(--text-color);
    background-color: var(--bg-color);
    margin: 0;
    padding: 0;
}

/* Layout */
header {
    background-color: var(--primary-color);
    color: white;
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

header h1 {
    margin: 0;
    font-size: 1.8rem;
}

nav a {
    color: white;
    text-decoration: none;
    margin-left: 1rem;
}

nav a:hover {
    text-decoration: underline;
}

main {
    max-width: 1200px;
    margin: 2rem auto;
    padding: 0 2rem;
}

footer {
    background-color: var(--primary-color);
    color: white;
    text-align: center;
    padding: 1rem;
    margin-top: 3rem;
}

/* Content styling */
h2 {
    color: var(--primary-color);
    border-bottom: 2px solid var(--secondary-color);
    padding-bottom: 0.5rem;
}

h3 {
    color: var(--primary-color);
    margin-top: 2rem;
}

a {
    color: var(--secondary-color);
}

code {
    font-family: 'Courier New', Courier, monospace;
    background-color: var(--code-bg);
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
}

.module-path {
    font-family: 'Courier New', Courier, monospace;
    color: #777;
    margin-bottom: 1rem;
}

.module-contents {
    background-color: white;
    border: 1px solid var(--border-color);
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}

.module-contents ul {
    list-style-type: none;
    padding-left: 1rem;
}

.module-contents li {
    margin-bottom: 0.5rem;
}

/* Class and function styling */
.class, .function {
    background-color: white;
    border: 1px solid var(--border-color);
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}

.class-name, .function-name {
    color: var(--secondary-color);
    font-family: 'Courier New', Courier, monospace;
    margin-top: 0;
}

.method {
    border-top: 1px solid var(--border-color);
    padding-top: 1rem;
    margin-top: 1rem;
}

.method-name {
    font-family: 'Courier New', Courier, monospace;
    margin: 0;
}

.parameters, .returns {
    margin-top: 1rem;
}

.parameters h5, .returns h5, .parameters h6, .returns h6 {
    margin-bottom: 0.5rem;
}

.parameters ul {
    list-style-type: none;
    padding-left: 1rem;
}

.parameters li {
    margin-bottom: 0.5rem;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    header {
        flex-direction: column;
        align-items: flex-start;
    }
    
    nav {
        margin-top: 1rem;
    }
    
    nav a {
        margin-left: 0;
        margin-right: 1rem;
    }
    
    main {
        padding: 0 1rem;
    }
}
"""
        
        # Write CSS to file
        with open(os.path.join(self.output_dir, "css", "style.css"), "w") as f:
            f.write(css)

def main():
    """Main function to run the documentation generator"""
    parser = argparse.ArgumentParser(description="Generate documentation for Python code")
    parser.add_argument("source_dir", help="Directory containing source code")
    parser.add_argument("--output-dir", "-o", default="docs", help="Output directory for documentation")
    parser.add_argument("--name", "-n", default="Defect Detection System", help="Project name")
    args = parser.parse_args()
    
    # Check if source directory exists
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory {args.source_dir} does not exist")
        return 1
    
    # Create documentation generator
    generator = DocGenerator(args.source_dir, args.output_dir, args.name)
    
    # Generate documentation
    generator.generate_documentation()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())