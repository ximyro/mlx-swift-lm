// Copyright © 2025 Apple Inc.

import Foundation

/// Parses Kimi/Moonshot sections and individual calls with complete JSON arguments.
public struct KimiK2ToolCallParser: ToolCallParser {
    public let startTag: String? = "<|tool_calls_section_begin|>"
    public let endTag: String? = "<|tool_calls_section_end|>"
    public let startTags = [
        "<|tool_calls_section_begin|>", "<|tool_call_section_begin|>", "<|tool_call_begin|>",
    ]
    private let scanner = JSONToolCallFrameScanner(
        sections: [
            ("<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"),
            ("<|tool_call_section_begin|>", "<|tool_call_section_end|>"),
        ],
        callStart: "<|tool_call_begin|>", callEnd: "<|tool_call_end|>",
        allowsUnwrappedCall: true)

    public init() {}

    public func endTag(forStartTag startTag: String) -> String? {
        switch startTag {
        case "<|tool_calls_section_begin|>": return "<|tool_calls_section_end|>"
        case "<|tool_call_section_begin|>": return "<|tool_call_section_end|>"
        default: return "<|tool_call_end|>"
        }
    }

    public func frameEnd(in content: String) -> String.Index? { scanner.scan(content)?.end }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseCalls(content: content, tools: tools).first
    }

    public func parseEOS(_ content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        parseCalls(content: content, tools: tools)
    }

    public func parseCalls(content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        let text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let frame = scanner.scan(text) else {
            guard !startTags.contains(where: text.hasPrefix) else { return [] }
            return parseBody(text[...]).map { [$0] } ?? []
        }
        guard text[frame.end...].allSatisfy(\.isWhitespace) else { return [] }
        let calls = frame.bodies.compactMap(parseBody)
        return calls.count == frame.bodies.count ? calls : []
    }

    private func parseBody(_ body: Substring) -> ToolCall? {
        guard let argumentsStart = body.range(of: "<|tool_call_argument_begin|>") else {
            return nil
        }
        var name = body[..<argumentsStart.lowerBound].trimmingWhitespace()
        if let colon = name.lastIndex(of: ":") {
            let index = name[name.index(after: colon)...]
            guard !index.isEmpty, index.allSatisfy(\.isNumber) else { return nil }
            name = name[..<colon]
        }
        if let dot = name.firstIndex(of: ".") { name = name[name.index(after: dot)...] }
        guard !name.isEmpty, !name.contains(where: { $0.isWhitespace || "<>{}".contains($0) }),
            let arguments = tryParseJSON(String(body[argumentsStart.upperBound...]))
                as? [String: any Sendable]
        else { return nil }
        return ToolCall(function: .init(name: String(name), arguments: arguments))
    }
}
