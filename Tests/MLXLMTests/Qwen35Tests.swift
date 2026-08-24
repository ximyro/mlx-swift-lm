import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

extension MLXTestingSuite {
    @Suite
    struct Qwen35Tests {

        private func gatedDeltaReference(
            q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray,
            aLog: MLXArray, dtBias: MLXArray, initialState: MLXArray
        ) -> (MLXArray, MLXArray) {
            let repeatFactor = v.dim(2) / q.dim(2)
            let q = repeated(q, count: repeatFactor, axis: -2)
            let k = repeated(k, count: repeatFactor, axis: -2)
            let beta = sigmoid(b)
            let g = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
            var state = initialState
            var outputs = [MLXArray]()

            for t in 0 ..< q.dim(1) {
                state = state * expandedDimensions(g[0..., t], axes: [2, 3])
                let kv = (state * expandedDimensions(k[0..., t], axis: -2)).sum(axis: -1)
                let delta = (v[0..., t] - kv) * expandedDimensions(beta[0..., t], axis: -1)
                state = state
                    + expandedDimensions(k[0..., t], axis: -2)
                    * expandedDimensions(delta, axis: -1)
                outputs.append(
                    (state * expandedDimensions(q[0..., t], axis: -2)).sum(axis: -1))
            }

            return (stacked(outputs, axis: 1), state)
        }

        private func makeTinyConfigData() -> Data {
            let json = """
            {
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "rope_theta": 10000.0,
                "max_position_embeddings": 512
            }
            """
            return json.data(using: .utf8)!
        }

        @Test("Qwen35 callCapturing returns captured layers")
        func testQwen35CallCapturing() throws {

            let data = makeTinyConfigData()
            let config = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
            let model = Qwen35Model(config)

            let input = MLXArray(0..<8).reshaped(1, 8)
            let captureLayerIDs: Set<Int> = [0, 1]
            
            let (hiddenStates, captured) = model.languageModel.model.callCapturing(input, captureLayerIDs: captureLayerIDs)

            #expect(hiddenStates.shape == [1, 8, 64])
            #expect(captured.count == 2)
            #expect(captured[0]?.shape == [1, 8, 64])
            #expect(captured[1]?.shape == [1, 8, 64])
        }

        @Test("Shared Qwen3.5 GatedDelta matches the ops recurrence with cached state")
        func testGatedDeltaKernelMatchesReference() {
            let B = 1, T = 4, Hk = 1, Dk = 32, Hv = 2, Dv = 4
            let q = MLXArray(0 ..< (B * T * Hk * Dk)).asType(.float32)
                .reshaped(B, T, Hk, Dk) / 100
            let k = MLXArray(1 ... (B * T * Hk * Dk)).asType(.float32)
                .reshaped(B, T, Hk, Dk) / 200
            let v = MLXArray(0 ..< (B * T * Hv * Dv)).asType(.float32)
                .reshaped(B, T, Hv, Dv) / 50
            let a = MLXArray(1 ... (B * T * Hv)).asType(.float32)
                .reshaped(B, T, Hv) / 20
            let b = MLXArray(0 ..< (B * T * Hv)).asType(.float32)
                .reshaped(B, T, Hv) / 10
            let aLog = MLXArray([Float(-1), Float(-0.5)])
            let dtBias = MLXArray([Float(0.1), Float(0.2)])
            let state = MLXArray(0 ..< (B * Hv * Dv * Dk)).asType(.float32)
                .reshaped(B, Hv, Dv, Dk) / 1_000

            let expected = gatedDeltaReference(
                q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
                initialState: state)
            let actual = gatedDeltaUpdate(
                q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
                state: state)
            eval(expected.0, expected.1, actual.0, actual.1)

            #expect(allClose(actual.0, expected.0, rtol: 1e-4, atol: 1e-5).item(Bool.self))
            #expect(allClose(actual.1, expected.1, rtol: 1e-4, atol: 1e-5).item(Bool.self))
        }
    }
}
