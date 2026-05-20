// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for GPT-OSS Harmony tool calls.
///
/// Harmony tool calls are emitted on the commentary channel, commonly as:
/// `<|channel|>commentary to=functions.name <|constrain|>json<|message|>{...}<|call|>`.
/// The `<|call|>` token is often consumed as an EOS token, so `parseEOS` must accept
/// a call block without the closing marker.
public struct HarmonyToolCallParser: TaggedToolCallParser, Sendable {
    public let startTag: String? = "<|channel|>commentary"
    public let endTag: String? = "<|call|>"
    public let startTags: [String] = ["<|channel|>commentary", "to=functions."]

    public init() {}

    public func endTags(forStartTag startTag: String) -> [String] {
        ["<|call|>"]
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseCalls(in: content).first
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        parseCalls(in: toolCallBuffer)
    }

    private func parseCalls(in content: String) -> [ToolCall] {
        var calls: [ToolCall] = []
        var searchStart = content.startIndex

        while let functionRange = content.range(of: "to=functions.", range: searchStart..<content.endIndex) {
            let nameStart = functionRange.upperBound
            var nameEnd = nameStart
            while nameEnd < content.endIndex, isFunctionNameCharacter(content[nameEnd]) {
                nameEnd = content.index(after: nameEnd)
            }

            guard nameEnd > nameStart else {
                searchStart = functionRange.upperBound
                continue
            }

            guard let messageRange = content.range(of: "<|message|>", range: nameEnd..<content.endIndex) else {
                break
            }

            let argsStart = messageRange.upperBound
            let argsEnd = firstRange(
                of: ["<|call|>", "<|end|>", "<|return|>", "<|start|>"],
                in: content,
                range: argsStart..<content.endIndex
            )?.lowerBound ?? content.endIndex

            let name = String(content[nameStart..<nameEnd])
            let rawArguments = String(content[argsStart..<argsEnd])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if let arguments = parseArguments(rawArguments) {
                calls.append(ToolCall(function: .init(name: name, arguments: arguments)))
            }

            searchStart = argsEnd
        }

        return calls
    }

    private func parseArguments(_ raw: String) -> [String: any Sendable]? {
        guard !raw.isEmpty, let data = raw.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        else {
            return raw.isEmpty ? [:] : ["arguments": raw]
        }

        if let dict = object as? [String: Any] {
            return dict.mapValues(asSendable)
        }

        return ["arguments": asSendable(object)]
    }

    private func isFunctionNameCharacter(_ character: Character) -> Bool {
        character.isLetter || character.isNumber || character == "_" || character == "-"
    }

    private func firstRange(
        of needles: [String],
        in text: String,
        range: Range<String.Index>
    ) -> Range<String.Index>? {
        needles.compactMap { text.range(of: $0, range: range) }
            .min { $0.lowerBound < $1.lowerBound }
    }
}
