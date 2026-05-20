// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Kimi/Moonshot tool call format.
///
/// Supports both section-wrapped and bare tool calls:
/// `<|tool_calls_section_begin|>...<|tool_calls_section_end|>`
/// `<|tool_call_section_begin|>...<|tool_call_section_end|>`
/// `<|tool_call_begin|>functions.name:0<|tool_call_argument_begin|>{...}<|tool_call_end|>`
public struct KimiK2ToolCallParser: TaggedToolCallParser, Sendable {
    public let startTag: String? = "<|tool_calls_section_begin|>"
    public let endTag: String? = "<|tool_calls_section_end|>"

    public let startTags = [
        "<|tool_calls_section_begin|>",
        "<|tool_call_section_begin|>",
        "<|tool_call_begin|>",
    ]

    private let sectionPairs = [
        ("<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"),
        ("<|tool_call_section_begin|>", "<|tool_call_section_end|>"),
    ]

    private let callStart = "<|tool_call_begin|>"
    private let callEnd = "<|tool_call_end|>"
    private let argumentStart = "<|tool_call_argument_begin|>"

    public init() {}

    public func endTags(forStartTag startTag: String) -> [String] {
        switch startTag {
        case "<|tool_calls_section_begin|>":
            return ["<|tool_calls_section_end|>"]
        case "<|tool_call_section_begin|>":
            return ["<|tool_call_section_end|>"]
        case "<|tool_call_begin|>":
            return [callEnd]
        default:
            return endTag.map { [$0] } ?? []
        }
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseEOS(content, tools: tools).first
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        var text = toolCallBuffer
        var calls: [ToolCall] = []

        for (start, end) in sectionPairs {
            let blocks = extractDelimitedBlocks(from: text, start: start, end: end)
            for block in blocks {
                let parsed = parseCallBlocks(in: block.inner)
                if parsed.isEmpty, let call = parseSingleCall(block.inner) {
                    calls.append(call)
                } else {
                    calls.append(contentsOf: parsed)
                }
            }

            for block in blocks.reversed() {
                text.removeSubrange(block.fullRange)
            }
        }

        calls.append(contentsOf: parseCallBlocks(in: text))
        return calls
    }

    private func parseCallBlocks(in text: String) -> [ToolCall] {
        extractDelimitedBlocks(from: text, start: callStart, end: callEnd)
            .compactMap { parseSingleCall($0.inner) }
    }

    private func parseSingleCall(_ content: String) -> ToolCall? {
        var text = content
        for (start, end) in sectionPairs {
            text = text.replacingOccurrences(of: start, with: "")
            text = text.replacingOccurrences(of: end, with: "")
        }
        text = text.replacingOccurrences(of: callStart, with: "")
        text = text.replacingOccurrences(of: callEnd, with: "")
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let argBeginRange = text.range(of: argumentStart) else { return nil }

        let rawName = String(text[..<argBeginRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let funcName = normalizeFunctionName(rawName)
        guard !funcName.isEmpty else { return nil }

        let argsText = String(text[argBeginRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let parsedArguments = tryParseJSON(argsText) else { return nil }

        let arguments: [String: any Sendable]
        if let dict = parsedArguments as? [String: any Sendable] {
            arguments = dict
        } else {
            arguments = ["arguments": parsedArguments]
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    private func normalizeFunctionName(_ rawName: String) -> String {
        var name = rawName

        if let colonIndex = name.lastIndex(of: ":") {
            let suffixStart = name.index(after: colonIndex)
            let suffix = name[suffixStart...]
            if !suffix.isEmpty, suffix.allSatisfy(\.isNumber) {
                name = String(name[..<colonIndex])
            }
        }

        if let dotIndex = name.lastIndex(of: ".") {
            name = String(name[name.index(after: dotIndex)...])
        }

        return name.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func extractDelimitedBlocks(
        from text: String, start: String, end: String
    ) -> [(inner: String, fullRange: Range<String.Index>)] {
        var blocks: [(String, Range<String.Index>)] = []
        var searchStart = text.startIndex

        while let startRange = text.range(of: start, range: searchStart..<text.endIndex),
              let endRange = text.range(of: end, range: startRange.upperBound..<text.endIndex)
        {
            let fullRange = startRange.lowerBound..<endRange.upperBound
            blocks.append((String(text[startRange.upperBound..<endRange.lowerBound]), fullRange))
            searchStart = endRange.upperBound
        }

        return blocks
    }
}
