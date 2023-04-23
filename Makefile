.PHONY: run
run:
	ORT_DYLIB_PATH=`pwd`/target/debug/libonnxruntime.so cargo run
