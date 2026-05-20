// Copyright © 2025 Apple Inc.

import Foundation

/// Parses section-wrapped or individual DeepSeek V3/R1 calls with complete JSON arguments.
public struct DeepSeekToolCallParser: ToolCallParser {
    public let startTag: String? = "<｜tool▁calls▁begin｜>"
    public let endTag: String? = "<｜tool▁calls▁end｜>"
    public let startTags = ["<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>"]
    private let scanner = JSONToolCallFrameScanner(
        sections: [("<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>")],
        callStart: "<｜tool▁call▁begin｜>", callEnd: "<｜tool▁call▁end｜>",
        allowsJSONFence: true)

    public init() {}

    public func endTag(forStartTag startTag: String) -> String? {
        startTag == self.startTag ? endTag : "<｜tool▁call▁end｜>"
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
        guard let frame = scanner.scan(text), text[frame.end...].allSatisfy(\.isWhitespace) else {
            return []
        }
        let calls = frame.bodies.compactMap { body -> ToolCall? in
            guard body.hasPrefix("function<｜tool▁sep｜>"),
                let newline = body.firstIndex(of: "\n")
            else { return nil }
            let name = body.dropFirst("function<｜tool▁sep｜>".count)[..<newline]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty, !name.contains(where: \.isWhitespace) else { return nil }
            var json = body[body.index(after: newline)...].trimmingWhitespace()
            if json.hasPrefix("```json"), json.hasSuffix("```") {
                json = json.dropFirst(7).dropLast(3).trimmingWhitespace()
            }
            guard let arguments = tryParseJSON(String(json)) as? [String: any Sendable] else {
                return nil
            }
            return ToolCall(function: .init(name: name, arguments: arguments))
        }
        return calls.count == frame.bodies.count ? calls : []
    }
}
