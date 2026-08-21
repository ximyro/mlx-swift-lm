// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Qwen tool call formats.
///
/// Qwen checkpoints in the wild may emit OpenAI-style JSON wrappers,
/// bracket calls, or Qwen3.5/Qwen-Coder XML function calls.
public struct QwenToolCallParser: TaggedToolCallParser, Sendable {
    public let startTag: String? = "<tool_call>"
    public let endTag: String? = "</tool_call>"

    public let startTags = ["<tool_call>", "[Calling tool:", "<function="]

    public init() {}

    public func endTags(forStartTag startTag: String) -> [String] {
        switch startTag {
        case "<tool_call>":
            return ["</tool_call>"]
        case "[Calling tool:":
            return [")]"]
        case "<function=":
            return ["</function>"]
        default:
            return endTag.map { [$0] } ?? []
        }
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseEOS(content, tools: tools).first
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        var text = stripThinkTags(from: toolCallBuffer)
        var calls: [ToolCall] = []

        calls.append(contentsOf: parseBracketCalls(in: text))

        let toolCallBlocks = extractDelimitedBlocks(
            from: text, start: "<tool_call>", end: "</tool_call>")
        for block in toolCallBlocks {
            if let call = parseJSONToolCall(block.inner) {
                calls.append(call)
            } else {
                // A single <tool_call> wrapper may carry several <function=>
                // blocks (models batch calls); parse all of them, not just the
                // first.
                calls.append(contentsOf: parseXMLFunctionCalls(in: block.inner, tools: tools))
            }
        }

        for block in toolCallBlocks.reversed() {
            text.removeSubrange(block.fullRange)
        }

        calls.append(contentsOf: parseBareFunctionCalls(in: text, tools: tools))
        return calls
    }

    private func parseJSONToolCall(_ text: String) -> ToolCall? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("{") else { return nil }

        if let data = trimmed.data(using: .utf8),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let name = object["name"] as? String,
           !name.isEmpty
        {
            let rawArguments = object["arguments"] ?? [String: Any]()
            let arguments: [String: any Sendable]
            if let dict = rawArguments as? [String: Any] {
                arguments = dict.mapValues(asSendable)
            } else {
                arguments = ["arguments": asSendable(rawArguments)]
            }

            return ToolCall(function: .init(name: name, arguments: arguments))
        }

        // Lenient recovery: the block looked like JSON but didn't parse as a
        // whole (truncated tail, stray text after the object, …). Salvage the
        // name and the balanced-brace arguments object rather than dropping
        // the call or emitting it argument-less.
        guard let nameKey = trimmed.range(of: #""name""#),
              let colon = trimmed.range(of: ":", range: nameKey.upperBound ..< trimmed.endIndex),
              let quoteOpen = trimmed.range(of: "\"", range: colon.upperBound ..< trimmed.endIndex),
              let quoteClose = trimmed.range(
                of: "\"", range: quoteOpen.upperBound ..< trimmed.endIndex)
        else { return nil }

        let name = String(trimmed[quoteOpen.upperBound ..< quoteClose.lowerBound])
        guard !name.isEmpty else { return nil }

        var arguments: [String: any Sendable] = [:]
        if let argsKey = trimmed.range(of: #""arguments""#) {
            guard
                let brace = trimmed.range(
                    of: "{", range: argsKey.upperBound ..< trimmed.endIndex),
                let close = findBalancedBrace(in: trimmed, startingAt: brace.lowerBound),
                let data = String(trimmed[brace.lowerBound ... close]).data(using: .utf8),
                let dict = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            else { return nil }
            arguments = dict.mapValues(asSendable)
        }

        return ToolCall(function: .init(name: name, arguments: arguments))
    }

    /// Parse every `<function=…>` block in `text`, tolerating a missing
    /// `</function>` closer on the final block (the value then runs to the end
    /// of the wrapper the caller already stripped).
    private func parseXMLFunctionCalls(
        in text: String, tools: [[String: any Sendable]]?
    ) -> [ToolCall] {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        var calls: [ToolCall] = []
        var searchStart = text.startIndex
        while let start = text.range(of: "<function=", range: searchStart ..< text.endIndex) {
            let end = text.range(
                of: "</function>", range: start.upperBound ..< text.endIndex)
            let segmentEnd = end?.upperBound ?? text.endIndex
            let segment = String(text[start.lowerBound ..< segmentEnd])
            if let call = parser.parse(content: segment, tools: tools) {
                calls.append(call)
            }
            searchStart = segmentEnd
        }
        return calls
    }

    private func parseBracketCalls(in text: String) -> [ToolCall] {
        var calls: [ToolCall] = []
        var searchStart = text.startIndex

        while let marker = text.range(of: "[Calling tool:", range: searchStart..<text.endIndex) {
            var nameStart = marker.upperBound
            while nameStart < text.endIndex, text[nameStart].isWhitespace {
                nameStart = text.index(after: nameStart)
            }

            guard let openParen = text.range(of: "(", range: nameStart..<text.endIndex) else {
                break
            }

            let name = String(text[nameStart..<openParen.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else {
                searchStart = openParen.upperBound
                continue
            }

            guard let openBrace = text.range(of: "{", range: openParen.upperBound..<text.endIndex),
                  let closeBrace = findBalancedBrace(in: text, startingAt: openBrace.lowerBound)
            else {
                searchStart = openParen.upperBound
                continue
            }

            let afterBrace = text.index(after: closeBrace)
            guard text[afterBrace...].hasPrefix(")]") else {
                searchStart = afterBrace
                continue
            }

            let argsText = String(text[openBrace.lowerBound...closeBrace])
            if let args = tryParseJSON(argsText) as? [String: any Sendable] {
                calls.append(ToolCall(function: .init(name: name, arguments: args)))
            }

            searchStart = text.index(after: afterBrace)
        }

        return calls
    }

    private func parseBareFunctionCalls(
        in text: String, tools: [[String: any Sendable]]?
    ) -> [ToolCall] {
        parseXMLFunctionCalls(in: text, tools: tools)
    }

    private func stripThinkTags(from text: String) -> String {
        var result = text

        while let start = result.range(of: "<think>") {
            guard let end = result.range(of: "</think>", range: start.upperBound..<result.endIndex)
            else {
                result.removeSubrange(start.lowerBound..<result.endIndex)
                break
            }
            result.removeSubrange(start.lowerBound..<end.upperBound)
        }

        if let end = result.range(of: "</think>") {
            result = String(result[end.upperBound...])
        }

        return result
    }

    private func extractDelimitedBlocks(
        from text: String, start: String, end: String
    ) -> [(inner: String, fullText: String, fullRange: Range<String.Index>)] {
        var blocks: [(String, String, Range<String.Index>)] = []
        var searchStart = text.startIndex

        while let startRange = text.range(of: start, range: searchStart..<text.endIndex),
              let endRange = text.range(of: end, range: startRange.upperBound..<text.endIndex) {
            let fullRange = startRange.lowerBound..<endRange.upperBound
            blocks.append((
                String(text[startRange.upperBound..<endRange.lowerBound]),
                String(text[fullRange]),
                fullRange
            ))
            searchStart = endRange.upperBound
        }

        return blocks
    }

    private func findBalancedBrace(in text: String, startingAt start: String.Index) -> String.Index? {
        guard start < text.endIndex, text[start] == "{" else { return nil }

        var depth = 0
        var index = start
        var inString = false
        var escaped = false

        while index < text.endIndex {
            let char = text[index]

            if inString {
                if escaped {
                    escaped = false
                } else if char == "\\" {
                    escaped = true
                } else if char == "\"" {
                    inString = false
                }
            } else if char == "\"" {
                inString = true
            } else if char == "{" {
                depth += 1
            } else if char == "}" {
                depth -= 1
                if depth == 0 {
                    return index
                }
            }

            index = text.index(after: index)
        }

        return nil
    }
}
