// Copyright © 2025 Apple Inc.

import Foundation

/// Processes generated text to detect and extract tool calls during streaming generation.
///
/// `ToolCallProcessor` handles the streaming detection of tool calls in model output,
/// buffering partial content and extracting complete tool calls when detected.
///
/// Example:
/// ```swift
/// let processor = ToolCallProcessor(format: .lfm2)
/// for chunk in generatedChunks {
///     if let text = processor.processChunk(chunk) {
///         // Regular text to display
///         print(text)
///     }
/// }
/// // After generation completes:
/// for toolCall in processor.toolCalls {
///     // Handle extracted tool calls
///     print(toolCall.function.name)
/// }
/// ```
public class ToolCallProcessor {

    // MARK: - Properties

    private let parser: any ToolCallParser
    private let tools: [[String: any Sendable]]?
    private var state = State.normal
    private var toolCallBuffer = ""
    private var activeEndTags: [String] = []

    /// The tool calls extracted during processing.
    public var toolCalls: [ToolCall] = []

    // MARK: - State Enum

    private enum State {
        case normal
        case potentialToolCall
        case collectingToolCall
    }

    // MARK: - Initialization

    /// Initialize with a specific tool call format.
    /// - Parameters:
    ///   - format: The tool call format to use (defaults to `.json` for standard JSON format)
    ///   - tools: Optional tool schemas for type-aware parsing
    public init(format: ToolCallFormat = .json, tools: [[String: any Sendable]]? = nil) {
        self.parser = format.createParser()
        self.tools = tools
    }

    // MARK: - Computed Properties

    /// Whether this processor uses inline format (no start tag).
    private var isInlineFormat: Bool {
        parser.startTag == nil
    }

    /// The first character of the start tag for quick detection.
    private var startTagFirstChar: Character? {
        parser.startTag?.first
    }

    private var startTags: [String] {
        if let taggedParser = parser as? any TaggedToolCallParser {
            return taggedParser.startTags
        }
        return parser.startTag.map { [$0] } ?? []
    }

    private var startTagFirstChars: Set<Character> {
        Set(startTags.compactMap(\.first))
    }

    private func endTags(for startTag: String) -> [String] {
        if let taggedParser = parser as? any TaggedToolCallParser {
            return taggedParser.endTags(forStartTag: startTag)
        }
        return parser.endTag.map { [$0] } ?? []
    }

    // MARK: - Public Methods

    /// Process a generated text chunk and extract any tool call content.
    /// - Parameter chunk: The text chunk to process
    /// - Returns: Regular text that should be displayed (non-tool call content), or `nil` if buffering
    public func processChunk(_ chunk: String) -> String? {
        if isInlineFormat {
            return processInlineChunk(chunk)
        }
        return processTaggedChunk(chunk)
    }

    /// Process end-of-sequence, parsing any buffered content as tool call(s).
    ///
    /// Call this when generation ends (e.g., on EOS token) to handle formats
    /// whose end tag is never delivered as text (e.g., Mistral where `</s>`
    /// is intercepted at the token ID level).
    ///
    /// For formats with end tags that appear in the text stream, the buffer
    /// will already be empty at generation end, making this a no-op.
    public func processEOS() {
        guard state == .collectingToolCall || state == .potentialToolCall else { return }
        guard !toolCallBuffer.isEmpty else {
            state = .normal
            activeEndTags = []
            return
        }

        toolCalls.append(contentsOf: parser.parseEOS(toolCallBuffer, tools: tools))

        toolCallBuffer = ""
        activeEndTags = []
        state = .normal
    }

    // MARK: - Private Methods

    /// Process chunk for inline formats (no wrapper tags).
    ///
    /// Uses brace counting to detect when output looks like a JSON tool call.
    /// While braces are unbalanced the content is buffered (returns `nil`)
    /// so partial JSON is never leaked to the UI.
    private func processInlineChunk(_ chunk: String) -> String? {
        switch state {
        case .normal:
            // Check if this chunk starts what looks like a JSON tool call
            if let braceIndex = chunk.firstIndex(of: "{") {
                let leading = String(chunk[..<braceIndex])
                let jsonPart = String(chunk[braceIndex...])
                toolCallBuffer = jsonPart
                state = .collectingToolCall

                if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                    toolCalls.append(toolCall)
                    toolCallBuffer = ""
                    state = .normal
                    return leading.isEmpty ? nil : leading
                }

                // Still collecting — check if braces are balanced (would mean parse
                // failed on complete JSON, so it's not a tool call)
                if jsonBracesBalanced(toolCallBuffer) {
                    state = .normal
                    let buffer = toolCallBuffer
                    toolCallBuffer = ""
                    return leading + buffer
                }

                return leading.isEmpty ? nil : leading
            }

            // No brace seen — pass through as regular text
            return chunk

        case .potentialToolCall, .collectingToolCall:
            toolCallBuffer += chunk

            if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                toolCalls.append(toolCall)
                toolCallBuffer = ""
                state = .normal
                return nil
            }

            // If braces are balanced but parse failed, this isn't a tool call — flush
            if jsonBracesBalanced(toolCallBuffer) {
                state = .normal
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                return buffer
            }

            // Still collecting
            return nil
        }
    }

    /// Check whether open/close braces are balanced in the string.
    private func jsonBracesBalanced(_ text: String) -> Bool {
        var depth = 0
        for ch in text {
            if ch == "{" { depth += 1 } else if ch == "}" { depth -= 1 }
        }
        return depth == 0
    }

    /// Process chunk for tagged formats.
    private func processTaggedChunk(_ chunk: String) -> String? {
        let startTags = self.startTags
        guard !startTags.isEmpty else {
            return chunk
        }

        let firstChars = startTagFirstChars
        guard (state == .normal && chunk.contains { firstChars.contains($0) }) || state != .normal
        else {
            return chunk
        }

        toolCallBuffer += chunk
        var leadingToken: String?

        switch state {
        case .normal:
            if let startRange = firstRange(of: startTags, in: toolCallBuffer) {
                let startTag = String(toolCallBuffer[startRange])
                leadingToken = separateToken(
                    from: &toolCallBuffer, separatorRange: startRange, returnLeading: true)
                activeEndTags = endTags(for: startTag)
                state = .collectingToolCall
                fallthrough
            } else if let partialRange = trailingPartialStartTagRange(
                in: toolCallBuffer, tags: startTags)
            {
                leadingToken = String(toolCallBuffer[..<partialRange.lowerBound])
                toolCallBuffer = String(toolCallBuffer[partialRange.lowerBound...])
                state = .potentialToolCall
                return leadingToken?.isEmpty ?? true ? nil : leadingToken
            } else {
                state = .normal
                activeEndTags = []
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                return buffer
            }
        case .potentialToolCall:
            if partialMatch(buffer: toolCallBuffer, tags: startTags) {
                if let startTag = completedStartTag(in: toolCallBuffer, tags: startTags) {
                    activeEndTags = endTags(for: startTag)
                    state = .collectingToolCall
                    fallthrough
                } else {
                    return nil
                }
            } else {
                // Otherwise, return the collected text and reset the state
                state = .normal
                activeEndTags = []
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                return (leadingToken ?? "") + buffer
            }
        case .collectingToolCall:
            let endTags = activeEndTags.isEmpty ? parser.endTag.map { [$0] } ?? [] : activeEndTags
            guard !endTags.isEmpty else {
                return nil
            }

            if let endRange = firstRange(of: endTags, in: toolCallBuffer) {
                // Separate the trailing token
                let trailingToken = separateToken(
                    from: &toolCallBuffer, separatorRange: endRange, returnLeading: false)

                toolCalls.append(contentsOf: parser.parseEOS(toolCallBuffer, tools: tools))

                state = .normal
                activeEndTags = []
                toolCallBuffer = ""

                // If the token contains the start character, there may be more tool calls to come
                let nextOutput: String?
                if let trailingToken,
                    trailingToken.contains(where: { firstChars.contains($0) })
                {
                    nextOutput = processChunk(trailingToken)
                } else {
                    // Otherwise, return the collected token, or nil if it's empty
                    nextOutput = trailingToken?.isEmpty ?? true ? nil : trailingToken
                }
                
                if let leading = leadingToken, !leading.isEmpty {
                    return leading + (nextOutput ?? "")
                }
                return nextOutput
            } else {
                return nil
            }
        }
    }

    /// Separates a token from a string buffer based on a separator
    /// - Parameters:
    ///   - buffer: The string buffer to modify
    ///   - separator: The separator string to search for
    ///   - returnLeading: If true, returns text before separator; if false, returns text after
    /// - Returns: The separated token, or nil if separator not found
    private func separateToken(from buffer: inout String, separator: String, returnLeading: Bool)
        -> String?
    {
        guard let range = buffer.range(of: separator) else { return nil }

        let token: String
        if returnLeading {
            token = String(buffer[..<range.lowerBound])
            buffer = String(buffer[range.lowerBound...])
        } else {
            token = String(buffer[range.upperBound...])
            buffer = String(buffer[..<range.upperBound])
        }

        return token
    }

    private func separateToken(
        from buffer: inout String, separators: Set<Character>, returnLeading: Bool
    ) -> String? {
        guard let index = buffer.firstIndex(where: { separators.contains($0) }) else { return nil }

        let token: String
        if returnLeading {
            token = String(buffer[..<index])
            buffer = String(buffer[index...])
        } else {
            token = String(buffer[buffer.index(after: index)...])
            buffer = String(buffer[...index])
        }

        return token
    }

    private func separateToken(
        from buffer: inout String, separatorRange: Range<String.Index>, returnLeading: Bool
    ) -> String? {
        let token: String
        if returnLeading {
            token = String(buffer[..<separatorRange.lowerBound])
            buffer = String(buffer[separatorRange.lowerBound...])
        } else {
            token = String(buffer[separatorRange.upperBound...])
            buffer = String(buffer[..<separatorRange.upperBound])
        }

        return token
    }

    private func completedStartTag(in buffer: String, tags: [String]) -> String? {
        tags.first { buffer.hasPrefix($0) }
    }

    private func partialMatch(buffer: String, tags: [String]) -> Bool {
        tags.contains { tag in
            for (tagIndex, bufferIndex) in zip(tag.indices, buffer.indices) {
                if buffer[bufferIndex] != tag[tagIndex] {
                    return false
                }
            }
            return true
        }
    }

    private func firstRange(of tags: [String], in text: String) -> Range<String.Index>? {
        tags
            .compactMap { text.range(of: $0) }
            .min { lhs, rhs in lhs.lowerBound < rhs.lowerBound }
    }

    private func trailingPartialStartTagRange(in text: String, tags: [String]) -> Range<String.Index>? {
        guard !text.isEmpty else { return nil }

        for index in text.indices {
            let suffix = String(text[index...])
            guard !suffix.isEmpty else { continue }
            if tags.contains(where: { $0.hasPrefix(suffix) && suffix.count < $0.count }) {
                return index..<text.endIndex
            }
        }

        return nil
    }
}
