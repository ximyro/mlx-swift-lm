// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for XML function format: <function=name><parameter=key>value</parameter></function>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/qwen3_coder.py
public struct XMLFunctionParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    public init(startTag: String, endTag: String) {
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Lenient span extraction: quantized models at long context sometimes
        // omit the closing </function> tag. Take everything from <function= to
        // </function> when present, else to the end of the content — the caller
        // already bounds `content` at the wrapper's end tag (e.g. </tool_call>).
        guard let funcStart = content.range(of: "<function=") else { return nil }
        let funcEnd = content.range(
            of: "</function>", range: funcStart.upperBound ..< content.endIndex)
        let funcContent = String(
            content[funcStart.lowerBound ..< (funcEnd?.lowerBound ?? content.endIndex)])

        // Extract function name (everything between <function= and first >)
        guard let nameStart = funcContent.range(of: "<function="),
            let nameEnd = funcContent.range(
                of: ">", range: nameStart.upperBound ..< funcContent.endIndex)
        else { return nil }

        let funcName = String(funcContent[nameStart.upperBound ..< nameEnd.lowerBound])
        let paramSection = String(funcContent[nameEnd.upperBound...])

        var arguments: [String: any Sendable] = [:]

        // Find all parameter tags
        var searchRange = paramSection.startIndex ..< paramSection.endIndex
        while let paramStart = paramSection.range(of: "<parameter=", range: searchRange) {
            // Find the parameter name (between = and >)
            guard
                let nameEnd = paramSection.range(
                    of: ">", range: paramStart.upperBound ..< paramSection.endIndex)
            else { break }

            let paramName = String(paramSection[paramStart.upperBound ..< nameEnd.lowerBound])

            // Find the value's end: the closing </parameter> tag when present.
            // Models sometimes omit it — recover by ending the value at the next
            // <parameter= (or the end of the function block) instead of dropping
            // this and all following arguments.
            let closer = paramSection.range(
                of: "</parameter>", range: nameEnd.upperBound ..< paramSection.endIndex)
            let nextParam = paramSection.range(
                of: "<parameter=", range: nameEnd.upperBound ..< paramSection.endIndex)
                .flatMap {
                    isAtLineBoundary($0.lowerBound, in: paramSection) ? $0 : nil
                }

            let valueEnd: String.Index
            let resumeFrom: String.Index
            if let closer, nextParam == nil || closer.lowerBound < nextParam!.lowerBound {
                valueEnd = closer.lowerBound
                resumeFrom = closer.upperBound
            } else if let nextParam {
                valueEnd = nextParam.lowerBound
                resumeFrom = nextParam.lowerBound
            } else {
                valueEnd = paramSection.endIndex
                resumeFrom = paramSection.endIndex
            }

            var paramValue = String(paramSection[nameEnd.upperBound ..< valueEnd])

            // Trim leading/trailing newlines (matching Python behavior)
            if paramValue.hasPrefix("\n") {
                paramValue = String(paramValue.dropFirst())
            }
            if paramValue.hasSuffix("\n") {
                paramValue = String(paramValue.dropLast())
            }

            // Convert value based on schema type. Without a schema, mirror Qwen
            // parser behavior by accepting JSON literals before falling back to text.
            if getParameterType(funcName: funcName, paramName: paramName, tools: tools) != nil {
                arguments[paramName] = convertParameterValue(
                    paramValue, paramName: paramName, funcName: funcName, tools: tools)
            } else {
                arguments[paramName] = tryParseJSON(paramValue) ?? paramValue
            }

            searchRange = resumeFrom ..< paramSection.endIndex
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    private func isAtLineBoundary(_ index: String.Index, in text: String) -> Bool {
        var cursor = index
        while cursor > text.startIndex {
            let preceding = text.index(before: cursor)
            let character = text[preceding]
            if character == "\n" || character == "\r" {
                return true
            }
            if !character.isWhitespace {
                return false
            }
            cursor = preceding
        }
        return true
    }
}
