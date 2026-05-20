// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for DeepSeek V3/R1 tool call format.
///
/// DeepSeek uses full-width and underscore special tokens:
/// `<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>`
/// `<｜tool▁call▁begin｜>function<｜tool▁sep｜>name\n```json\n{...}\n```<｜tool▁call▁end｜>`
public struct DeepSeekToolCallParser: TaggedToolCallParser, Sendable {
    public let startTag: String? = "<｜tool▁calls▁begin｜>"
    public let endTag: String? = "<｜tool▁calls▁end｜>"

    public let startTags = [
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>",
    ]

    private let callsStart = "<｜tool▁calls▁begin｜>"
    private let callsEnd = "<｜tool▁calls▁end｜>"
    private let callStart = "<｜tool▁call▁begin｜>"
    private let callEnd = "<｜tool▁call▁end｜>"
    private let toolSeparator = "<｜tool▁sep｜>"

    public init() {}

    public func endTags(forStartTag startTag: String) -> [String] {
        switch startTag {
        case callsStart:
            return [callsEnd]
        case callStart:
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

        let sectionBlocks = extractDelimitedBlocks(from: text, start: callsStart, end: callsEnd)
        for block in sectionBlocks {
            calls.append(contentsOf: parseCallBlocks(in: block.inner))
        }

        for block in sectionBlocks.reversed() {
            text.removeSubrange(block.fullRange)
        }

        calls.append(contentsOf: parseCallBlocks(in: text))
        return calls
    }

    private func parseCallBlocks(in text: String) -> [ToolCall] {
        extractDelimitedBlocks(from: text, start: callStart, end: callEnd)
            .compactMap { parseSingleCall($0.inner) }
    }

    private func parseSingleCall(_ content: String) -> ToolCall? {
        let text = content.trimmingCharacters(in: .whitespacesAndNewlines)

        let nameAndBody: (name: String, body: String)
        if let separatorRange = text.range(of: toolSeparator) {
            let bodyStart = separatorRange.upperBound
            let nameAndRemainder = String(text[bodyStart...])
            nameAndBody = splitNameAndBody(nameAndRemainder)
        } else {
            nameAndBody = splitNameAndBody(text)
        }

        let name = nameAndBody.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }

        let argsText = extractJSONFence(from: nameAndBody.body)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !argsText.isEmpty else { return nil }

        let arguments: [String: any Sendable]
        if let parsed = tryParseJSON(argsText) {
            if let dict = parsed as? [String: any Sendable] {
                arguments = dict
            } else {
                arguments = ["arguments": parsed]
            }
        } else {
            arguments = ["arguments": argsText]
        }

        return ToolCall(function: .init(name: name, arguments: arguments))
    }

    private func splitNameAndBody(_ text: String) -> (name: String, body: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let newline = trimmed.firstIndex(of: "\n") else {
            return (trimmed, "")
        }
        return (
            String(trimmed[..<newline]),
            String(trimmed[trimmed.index(after: newline)...])
        )
    }

    private func extractJSONFence(from text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let fenceStart = trimmed.range(of: "```json") else {
            return trimmed
        }

        let jsonStart = fenceStart.upperBound
        guard let fenceEnd = trimmed.range(of: "```", range: jsonStart..<trimmed.endIndex) else {
            return String(trimmed[jsonStart...])
        }

        return String(trimmed[jsonStart..<fenceEnd.lowerBound])
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
