"""배포 파일 목록 self-check (표준 라이브러리만): PYTHONPATH=. python tests/test_deploy_files.py
서버 브리지가 import하는 src.core 모듈(전이적으로)이 deploy/build_package.sh와 deploy/push_model.sh --code 목록에 모두 있는지, 두 목록이 같은지.
"""
import re

ROOTS = ("__init__", "bridge_server", "model")
IMPORT = re.compile(r"^\s*from src\.core\.(\w+) import|^\s*import src\.core\.(\w+)", re.M)


def required():
    need, todo = set(), list(ROOTS)
    while todo:
        mod = todo.pop()
        if mod in need:
            continue
        need.add(mod)
        src = open(f"src/core/{mod}.py", encoding="utf-8").read()
        todo += [a or b for a, b in IMPORT.findall(src)]
    return need


def listed():
    build = re.search(r"for f in ([\w ]+); do cp src/core", open("deploy/build_package.sh", encoding="utf-8").read()).group(1).split()
    push = re.search(r"src/core/\{([\w,]+)\}\.py", open("deploy/push_model.sh", encoding="utf-8").read()).group(1).split(",")
    return set(build), set(push)


def test_deploy_lists():
    need = required()
    build, push = listed()
    assert build == push, f"두 배포 목록이 다름: build만 {sorted(build - push)}, push만 {sorted(push - build)}"
    assert need <= build, f"배포 목록에 빠진 모듈: {sorted(need - build)}"


if __name__ == "__main__":
    test_deploy_lists()
    print("ok")
