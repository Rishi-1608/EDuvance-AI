import ast
import os

src_file = "main_v7-9.py"
with open(src_file, "r", encoding="utf-8") as f:
    source = f.read()

lines = source.split("\n")
tree = ast.parse(source)

def extract_node_lines(node):
    start = node.lineno - 1
    if hasattr(node, 'decorator_list') and node.decorator_list:
        start = node.decorator_list[0].lineno - 1
    end = node.end_lineno
    return start, end

# Find where imports end (at `startup_event`)
imports_end = 0
for node in tree.body:
    if getattr(node, 'name', '') == "startup_event":
        start, end = extract_node_lines(node)
        imports_end = start
        break

imports_content_list = []
for line in lines[:imports_end]:
    if not line.strip().startswith("@app.on_event"):
        imports_content_list.append(line)
imports_content = "\n".join(imports_content_list)

route_groups = {
    'auth': ['register', 'login', 'me'],
    'upload': ['upload_video', 'upload_image', 'upload_audio', 'upload_document'],
    'generation': ['generate_flashcards', 'flashcard_generation_status'],
    'results': ['clear_results', 'status', 'results_video', 'results_image', 'results_audio', 'results_notes', 'results_pdf', 'results_flashcards', 'results_quiz', 'results_graph', 'results_frames', 'latest'],
    'media': ['stream_video', 'serve_minio_image'],
    'student': ['dashboard_stats', 'review_flashcard', 'get_progress', 'get_due_cards'],
    'system': ['probe_v1', 'delete_lecture', 'list_languages', 'stop_pipeline', 'root', 'diagnostics']
}

routes = []
route_lines = set()

for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        is_route = False
        if hasattr(node, 'decorator_list'):
            for d in node.decorator_list:
                if isinstance(d, ast.Call) and getattr(d.func, 'value', None) and getattr(d.func.value, 'id', None) == 'app':
                    is_route = True
                    break
        
        if is_route:
            start, end = extract_node_lines(node)
            routes.append({'name': node.name, 'start': start, 'end': end})
            for i in range(start, end):
                route_lines.add(i)

# Identify missing routes that didn't fit into groups
grouped_names = set(sum(route_groups.values(), []))
for r in routes:
    if r['name'] not in grouped_names:
        route_groups['system'].append(r['name'])

# Create structure
os.makedirs("app/api/endpoints", exist_ok=True)
os.makedirs("app/core", exist_ok=True)
os.makedirs("app/services", exist_ok=True)

# To generate __all__, find all top-level assignments, functions, and classes (excluding routes & startup)
exports = []
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if getattr(node, 'name', '') != 'startup_event' and node.name not in [r['name'] for r in routes]:
            exports.append(node.name)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                exports.append(target.id)

all_names_str = ",\n    ".join(f"'{name}'" for name in exports)
all_export_statement = f"__all__ = [\n    {all_names_str}\n]\n\n"

# Engine file: everything from `imports_end` down to the end of the file, skipping route lines and `startup_event`.
engine_file_content = imports_content + "\n\n" + all_export_statement

startup_node = next((n for n in tree.body if getattr(n, 'name', '') == 'startup_event'), None)
startup_lines_set = set()
if startup_node:
    s, e = extract_node_lines(startup_node)
    for i in range(s, e):
        startup_lines_set.add(i)

for i in range(imports_end, len(lines)):
    if i not in route_lines and i not in startup_lines_set:
        engine_file_content += lines[i] + "\n"

with open("app/core/engine.py", "w", encoding="utf-8") as f:
    f.write(engine_file_content)

router_template = """{imports}
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse, RedirectResponse
from app.core.engine import *

router = APIRouter()

"""

for group, r_list in route_groups.items():
    with open(f"app/api/endpoints/{group}.py", "w", encoding="utf-8") as f:
        f.write(router_template.replace("{imports}", imports_content))
        for r_name in r_list:
            r_data = next((r for r in routes if r['name'] == r_name), None)
            if r_data:
                code_lines = lines[r_data['start']:r_data['end']]
                code = "\n".join(code_lines)
                code = code.replace("@app.", "@router.")
                f.write(code + "\n\n")

startup_code = ""
if startup_node:
    s, e = extract_node_lines(startup_node)
    startup_code = "\n".join(lines[s:e])

main_code = f"""{imports_content}
from app.core.engine import *
from app.api.endpoints import auth, upload, generation, results, media, student, system

{startup_code}

app.include_router(auth.router, tags=["Authentication"])
app.include_router(upload.router, tags=["Uploads"])
app.include_router(generation.router, tags=["Generation"])
app.include_router(results.router, tags=["Results"])
app.include_router(media.router, tags=["Media"])
app.include_router(student.router, tags=["Student"])
app.include_router(system.router, tags=["System"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
"""

with open("app/main.py", "w", encoding="utf-8") as f:
    f.write(main_code)

print("Modularization complete! Files generated in app/ folder.")
