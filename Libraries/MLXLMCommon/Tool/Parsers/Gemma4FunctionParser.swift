// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Gemma 4 format: <|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>
/// Reference: Gemma 4 native format
public struct Gemma4FunctionParser: ToolCallParser, Sendable {
    public let startTag: String? = "<|tool_call>"
    public let endTag: String? = "<tool_call|>"

    private let escapeMarker = "<|\"|>"
    private let maxArgumentBlockLength = 1_048_576

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseCalls(in: content).first
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        parseCalls(in: toolCallBuffer)
    }

    private func parseCalls(in content: String) -> [ToolCall] {
        let text = stripThinkTags(from: content)
        let block = extractToolCallBlock(from: text)
        var calls: [ToolCall] = []
        var searchStart = block.startIndex

        while let callRange = block.range(of: "call:", range: searchStart..<block.endIndex) {
            var nameStart = callRange.upperBound
            while nameStart < block.endIndex, block[nameStart].isWhitespace {
                nameStart = block.index(after: nameStart)
            }

            var nameEnd = nameStart
            while nameEnd < block.endIndex, isFunctionNameCharacter(block[nameEnd]) {
                nameEnd = block.index(after: nameEnd)
            }

            guard nameEnd > nameStart else {
                searchStart = callRange.upperBound
                continue
            }

            var braceStart = nameEnd
            while braceStart < block.endIndex, block[braceStart].isWhitespace {
                braceStart = block.index(after: braceStart)
            }
            guard braceStart < block.endIndex, block[braceStart] == "{" else {
                searchStart = nameEnd
                continue
            }

            guard let braceEnd = findBalancedBrace(in: block, startingAt: braceStart) else {
                searchStart = block.index(after: braceStart)
                continue
            }

            let funcName = String(block[nameStart..<nameEnd])
            let argsRaw = String(block[braceStart...braceEnd])

            if let arguments = parseArguments(argsRaw) {
                calls.append(ToolCall(function: .init(name: funcName, arguments: arguments)))
            }

            searchStart = block.index(after: braceEnd)
        }

        return calls
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
        return result
    }

    private func extractToolCallBlock(from text: String) -> String {
        guard let startTag, let start = text.range(of: startTag) else {
            var stripped = text
            if let endTag {
                stripped = stripped.replacingOccurrences(of: endTag, with: "")
            }
            return stripped
        }

        let blockStart = start.upperBound
        if let endTag, let end = text.range(of: endTag, range: blockStart..<text.endIndex) {
            return String(text[blockStart..<end.lowerBound])
        }
        return String(text[blockStart...])
    }

    private func findBalancedBrace(in text: String, startingAt start: String.Index) -> String.Index? {
        guard text[start] == "{" else { return nil }
        if text.distance(from: start, to: text.endIndex) > maxArgumentBlockLength {
            return nil
        }

        var depth = 0
        var index = start
        var inEscapedString = false

        while index < text.endIndex {
            if text[index...].hasPrefix(escapeMarker) {
                inEscapedString.toggle()
                index = text.index(index, offsetBy: escapeMarker.count)
                continue
            }

            if !inEscapedString {
                if text[index] == "{" {
                    depth += 1
                } else if text[index] == "}" {
                    depth -= 1
                    if depth == 0 {
                        return index
                    }
                }
            }

            index = text.index(after: index)
        }

        return nil
    }

    private func parseArguments(_ raw: String) -> [String: any Sendable]? {
        let jsonText = gemma4ArgumentsToJSON(raw)
        guard let data = jsonText.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let dict = object as? [String: Any]
        else {
            return nil
        }

        return dict.mapValues(asSendable)
    }

    private func gemma4ArgumentsToJSON(_ raw: String) -> String {
        var strings: [String] = []
        var text = extractEscapedStrings(from: raw, into: &strings)
        text = replaceRegex(#"(?<=[{,])\s*(\w+)\s*:"#, in: text) { match, text in
            guard let keyRange = Range(match.range(at: 1), in: text) else { return nil }
            return "\"\(String(text[keyRange]))\":"
        }
        text = replaceRegex(#"(?<=[:\[,])(\s*)([A-Za-z_][\w\-]*)(?=\s*[,}\]])"#, in: text) {
            match, text in
            guard let wsRange = Range(match.range(at: 1), in: text),
                  let wordRange = Range(match.range(at: 2), in: text)
            else { return nil }

            let word = String(text[wordRange])
            if ["true", "false", "null"].contains(word) {
                return String(text[Range(match.range, in: text)!])
            }
            return String(text[wsRange]) + jsonStringLiteral(word)
        }

        return replaceRegex(#"@@(\d+)@@"#, in: text) { match, text in
            guard let idxRange = Range(match.range(at: 1), in: text),
                  let idx = Int(text[idxRange]),
                  strings.indices.contains(idx)
            else { return nil }
            return jsonStringLiteral(strings[idx])
        }
    }

    private func extractEscapedStrings(from text: String, into strings: inout [String]) -> String {
        var result = ""
        var index = text.startIndex

        while index < text.endIndex {
            guard text[index...].hasPrefix(escapeMarker) else {
                result.append(text[index])
                index = text.index(after: index)
                continue
            }

            let valueStart = text.index(index, offsetBy: escapeMarker.count)
            guard let valueEnd = text.range(of: escapeMarker, range: valueStart..<text.endIndex)
            else {
                result.append(contentsOf: text[index...])
                break
            }

            strings.append(String(text[valueStart..<valueEnd.lowerBound]))
            result += "@@\(strings.count - 1)@@"
            index = valueEnd.upperBound
        }

        return result
    }

    private func replaceRegex(
        _ pattern: String,
        in text: String,
        transform: (NSTextCheckingResult, String) -> String?
    ) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        var result = text
        let matches = regex.matches(
            in: text,
            range: NSRange(text.startIndex..<text.endIndex, in: text)
        )

        for match in matches.reversed() {
            guard let range = Range(match.range, in: result) else { continue }
            let replacement = transform(match, result) ?? String(result[range])
            result.replaceSubrange(range, with: replacement)
        }

        return result
    }

    private func jsonStringLiteral(_ value: String) -> String {
        guard let data = try? JSONEncoder().encode(value),
              let encoded = String(data: data, encoding: .utf8)
        else {
            return "\"\(value)\""
        }
        return encoded
    }

    private func isFunctionNameCharacter(_ char: Character) -> Bool {
        char.isLetter || char.isNumber || char == "_"
    }
}
