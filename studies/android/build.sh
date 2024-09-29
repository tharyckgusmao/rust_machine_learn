cargo ndk -t x86_64 -o ./jniLibs build --release
mkdir app/src/main/jniLibs
cp -r jniLibs app/src/main