// Copyright © 2026 Apple Inc.

import Foundation

struct QwenBracketToolCallParser: ToolCallParser {
    let startTag: String? = "[Calling tool:"
    let endTag: String? = ")]"
    private let scanner = StructuredTextScanner(quotes: ["\""])

    func frameEnd(in text: String) -> String.Index? {
        guard let startTag, text.hasPrefix(startTag),
            let open = scanner.firstTopLevelIndex(
                of: "(", in: text.dropFirst(startTag.count)),
            let close = scanner.endOfGroup(in: text[...], openedAt: open)
        else { return nil }
        let bracket = text.index(after: close)
        guard bracket < text.endIndex, text[bracket] == "]" else { return nil }
        return text.index(after: bracket)
    }

    func toolName(in text: String) -> String? {
        guard let startTag, text.hasPrefix(startTag), let open = text.firstIndex(of: "(") else {
            return nil
        }
        let name = text[text.index(text.startIndex, offsetBy: startTag.count) ..< open]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return name.isEmpty || name.contains(where: \.isWhitespace) ? nil : name
    }

    func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        let text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard frameEnd(in: text) == text.endIndex,
            let name = toolName(in: text), let open = text.firstIndex(of: "("),
            let close = scanner.endOfGroup(in: text[...], openedAt: open),
            let arguments = tryParseJSON(String(text[text.index(after: open) ..< close]))
                as? [String: any Sendable]
        else { return nil }
        return ToolCall(function: .init(name: name, arguments: arguments))
    }
}
