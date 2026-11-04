// Copyright © 2026 Apple Inc.
//
// Text-only loader for Prism ML's Bonsai 2 pack (model_type "prism_hadamard_qwen35",
// e.g. prism-ml/Ternary-Bonsai-2-27B-mlx-2bit). Ported from Rapid-MLX's
// rapid_mlx/models/prism_hadamard_qwen35.py, itself adapted from the pack's runtime/.
//
// The pack is an ordinary Qwen3.5 model in MLX tensor layout, except that every packed
// module listed in config.json `modules` stores affine 2-bit/g128 weights folded with a
// signed, normalized Walsh-Hadamard rotation of their input dimension. Those linears
// rotate activations before the quantized matmul; the embedding inverse-rotates rows
// after dequantization. Loading the pack as a plain quantized Qwen3.5 omits the
// rotations and produces noise.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct PrismHadamardLoadError: LocalizedError {
    let message: String
    var errorDescription: String? { "Bonsai 2 pack: \(message)" }
}

private func prismHadamard(_ x: MLXArray, block: Int, signs: MLXArray, inverse: Bool)
    -> MLXArray
{
    let shape = x.shape
    var y = x.asType(.float32)
    if !inverse { y = y * signs }
    y = hadamardTransform(y.reshaped(-1, block), scale: 1 / Float(block).squareRoot())
        .reshaped(shape)
    if inverse { y = y * signs }
    return y.asType(x.dtype)
}

/// Packed linear whose input is Hadamard-rotated before the 2-bit matmul.
///
/// A `QuantizedLinear` subclass so it fits `Linear` slots, while the fused-projection and
/// vocabulary fast paths (which match the exact `QuantizedLinear` type) leave it alone.
final class PrismHadamardLinear: QuantizedLinear {
    let block: Int
    let signs: MLXArray

    init(replacing other: Linear, block: Int) {
        let (rows, width) = other.shape
        self.block = block
        self.signs = MLXArray.ones([width], dtype: .float32)
        super.init(
            weight: MLXArray.zeros([rows, width / 16], dtype: .uint32), bias: nil,
            scales: MLXArray.zeros([rows, width / 128], dtype: .float16),
            biases: MLXArray.zeros([rows, width / 128], dtype: .float16),
            groupSize: 128, bits: 2)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(prismHadamard(x, block: block, signs: signs, inverse: false))
    }
}

/// Packed embedding whose dequantized rows are inverse-rotated.
final class PrismHadamardEmbedding: QuantizedEmbedding {
    let block: Int
    let signs: MLXArray

    init(replacing other: Embedding, block: Int) {
        let (rows, width) = other.shape
        self.block = block
        self.signs = MLXArray.ones([width], dtype: .float32)
        super.init(
            weight: MLXArray.zeros([rows, width / 16], dtype: .uint32),
            scales: MLXArray.zeros([rows, width / 128], dtype: .float16),
            biases: MLXArray.zeros([rows, width / 128], dtype: .float16),
            groupSize: 128, bits: 2)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        prismHadamard(
            super.callAsFunction(x).asType(.float16), block: block, signs: signs, inverse: true)
    }

    override func asLinear(_ x: MLXArray) -> MLXArray {
        fatalError("Bonsai 2 packs have an untied lm_head; a rotated embedding is not a head")
    }
}

public struct PrismHadamardQwen35Configuration: Codable, Sendable {
    struct PackedModule: Codable, Sendable {
        let path: String
        let block: Int
        let embedding: Bool
        let dtype: String
    }

    struct Quantization: Codable, Sendable, Equatable {
        let bits: Int
        let groupSize: Int
        let mode: String

        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
            case mode
        }
    }

    let base: Qwen35Configuration
    let modules: [PackedModule]
    let schemaVersion: Int
    let baseModelType: String
    let tensorNamespace: String
    let gdnActivationLayout: String
    let quantization: Quantization

    enum CodingKeys: String, CodingKey {
        case modules
        case schemaVersion = "schema_version"
        case baseModelType = "base_model_type"
        case tensorNamespace = "tensor_namespace"
        case gdnActivationLayout = "gdn_activation_layout"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modules = try c.decode([PackedModule].self, forKey: .modules)
        schemaVersion = try c.decode(Int.self, forKey: .schemaVersion)
        baseModelType = try c.decode(String.self, forKey: .baseModelType)
        tensorNamespace = try c.decode(String.self, forKey: .tensorNamespace)
        gdnActivationLayout = try c.decode(String.self, forKey: .gdnActivationLayout)
        quantization = try c.decode(Quantization.self, forKey: .quantization)
        base = try Qwen35Configuration(from: decoder)
        // Same contract Rapid-MLX accepts; anything else needs a new port, not a guess.
        guard schemaVersion == 2, baseModelType == "qwen3_5",
            tensorNamespace == "mlx-vlm-qwen3_5", gdnActivationLayout == "grouped",
            quantization == Quantization(bits: 2, groupSize: 128, mode: "affine"),
            !modules.isEmpty
        else {
            throw PrismHadamardLoadError(
                message: "expected schema-v2 Qwen3.5 affine 2-bit/g128 with a module manifest")
        }
    }

    public func encode(to encoder: Encoder) throws {
        try base.encode(to: encoder)
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modules, forKey: .modules)
        try c.encode(schemaVersion, forKey: .schemaVersion)
        try c.encode(baseModelType, forKey: .baseModelType)
        try c.encode(tensorNamespace, forKey: .tensorNamespace)
        try c.encode(gdnActivationLayout, forKey: .gdnActivationLayout)
        try c.encode(quantization, forKey: .quantization)
    }
}

public final class PrismHadamardQwen35Model: Qwen35Model {
    /// Packed module path (relative to `language_model`) → rotated input width.
    // Set after super.init, which builds the module tree they are derived from.
    private var packedWidths = [String: Int]()
    private var manifestErrors = [String]()

    public init(_ args: PrismHadamardQwen35Configuration) {
        super.init(args.base)
        let leaves = Dictionary(
            languageModel.leafModules().flattened(), uniquingKeysWith: { first, _ in first })
        var updates = [(String, Module)]()
        var widths = [String: Int]()
        var errors = [String]()
        for record in args.modules {
            guard widths[record.path] == nil else {
                errors.append("duplicate module \(record.path)")
                continue
            }
            guard record.dtype == "float16", [512, 1024, 2048, 4096].contains(record.block)
            else {
                errors.append("unsupported dtype or block for \(record.path)")
                continue
            }
            let replacement: Module
            let width: Int
            switch leaves[record.path] {
            case let embedding as Embedding where record.embedding:
                width = embedding.shape.1
                replacement = PrismHadamardEmbedding(replacing: embedding, block: record.block)
            case let linear as Linear where !record.embedding:
                width = linear.shape.1
                replacement = PrismHadamardLinear(replacing: linear, block: record.block)
            default:
                errors.append(
                    "no matching \(record.embedding ? "embedding" : "linear") at \(record.path)")
                continue
            }
            guard width % 128 == 0, width % record.block == 0 else {
                errors.append("width \(width) does not fit block \(record.block) at \(record.path)")
                continue
            }
            widths[record.path] = width
            updates.append((record.path, replacement))
        }
        packedWidths = widths
        manifestErrors = errors
        languageModel.update(modules: ModuleChildren.unflattened(updates))
    }

    public override func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws
        -> ModelCheckpoint
    {
        if let first = manifestErrors.first {
            throw PrismHadamardLoadError(message: first)
        }
        let prepared = try super.prepareCheckpoint(checkpoint)
        for (path, width) in packedWidths {
            guard let signs = prepared.weights["language_model.\(path).signs"],
                signs.shape == [width],
                all((signs .== 1) .|| (signs .== -1)).item(Bool.self)
            else {
                throw PrismHadamardLoadError(message: "missing or invalid sign vector for \(path)")
            }
        }
        return prepared
    }
}
