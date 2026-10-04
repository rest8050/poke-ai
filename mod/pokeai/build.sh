#!/bin/bash
# PokeAI 브리지 모드 빌드: 테스트 서버의 jar들만으로 javac 컴파일 (Gradle 불필요). 결과 jar를 coble-test/mods에 복사.
# 사용: bash build.sh [install]   (install이면 테스트 서버 mods에도 복사 — 서버가 꺼져 있을 때만)
cd "$(dirname "$0")"
T=/c/Users/lsh/Desktop/coble-test
JDK="/c/Program Files/Eclipse Adoptium/jdk-21.0.11.10-hotspot/bin"
w() { cygpath -m "$1"; }   # javac.exe는 C:/... 형식 경로만 이해함
CP="$(w $T/.fabric/remappedJars/minecraft-1.21.1-0.18.4/server-intermediary.jar)"
for j in $T/mods/Cobblemon-fabric-*.jar $T/mods/rctapi-*.jar $T/libraries/net/fabricmc/fabric-loader/0.18.4/*.jar /tmp/kt/META-INF/jars/kotlin-stdlib-2.4.10.jar; do CP="$CP;$(w $j)"; done
for j in $(find $T/libraries -name "*.jar"); do CP="$CP;$(w $j)"; done
mkdir -p out
"$JDK/javac" --release 21 -proc:none -nowarn -cp "$CP" -d out src/pokeai/*.java || { echo "컴파일 실패"; exit 1; }
cp res/fabric.mod.json out/fabric.mod.json
"$JDK/jar" cf pokeai-0.1.0.jar -C out .
if [ "$1" = "install" ]; then cp pokeai-0.1.0.jar $T/mods/pokeai-0.1.0.jar; echo "테스트 서버 mods에 설치 → $T/mods/pokeai-0.1.0.jar"; fi
echo "빌드 완료 → $(pwd)/pokeai-0.1.0.jar"
