cargo ndk -t x86_64 -o ./jniLibs build --release
mkdir android/app/src/main/jniLibs
cp -r jniLibs android/app/src/main