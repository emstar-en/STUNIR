# STUNIR Mobile Target

Mobile platform binding emitter for STUNIR (iOS/Android).

## Overview

This emitter converts STUNIR IR to native mobile platform code with
Swift bindings for iOS and Kotlin bindings for Android.

## Usage

```bash
python emitter.py <ir.json> --output=<output_dir> [--platform=ios|android|both]
```

## Output Files

### iOS
- `ios/<Class>.swift` - Swift implementation
- `ios/Package.swift` - Swift Package Manager manifest

### Android
- `android/<Class>.kt` - Kotlin implementation
- `android/build.gradle.kts` - Gradle build configuration

### Cross-Platform
- `interface.json` - Platform-agnostic interface definition
- `manifest.json` - Deterministic file manifest
- `README.md` - Generated documentation

## Features

- Type-safe Swift/Kotlin bindings
- SPM and Gradle build integration
- Cross-platform interface specification
- Native idiom code generation

## Schema

`stunir.mobile.ios.v1` / `stunir.mobile.android.v1`
