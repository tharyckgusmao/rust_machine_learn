package com.example.androidrust

class NativeLib {
    init {
        System.loadLibrary("android")
    }

    private external fun example()

    fun run() {
        example()
    }
}