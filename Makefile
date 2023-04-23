.PHONY: run
run:
	ORT_DYLIB_PATH=`pwd`/target/debug/libonnxruntime.so cargo run

.PHONY: fmt
fmt:
	cargo fmt
	git add -u
	cargo clippy --fix --allow-staged

.PHONY: check-fmt
check-fmt:
	cargo fmt --check
	cargo clippy
