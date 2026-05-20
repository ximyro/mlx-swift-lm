import Foundation
import MLXLMCommon
import Testing

struct DeepSeekKimiToolParserTests {
    private let tools: [[String: any Sendable]] = ["first", "second"].map {
        ["function": ["name": $0] as [String: any Sendable]]
    }

    @Test("DeepSeek and Kimi support sections and bare calls at every streaming boundary")
    func streamingBoundaries() {
        let kimi =
            "<|tool_call_begin|>functions.first:0<|tool_call_argument_begin|>{}<|tool_call_end|>"
        let deepseek =
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>first\n```json\n{}\n```<｜tool▁call▁end｜>"
        let cases: [(ToolCallFormat, String)] = [
            (.kimiK2, kimi),
            (.kimiK2, "<|tool_calls_section_begin|>\(kimi)<|tool_calls_section_end|>"),
            (.kimiK2, "<|tool_call_section_begin|>\(kimi)<|tool_call_section_end|>"),
            (.deepseek, deepseek),
            (.deepseek, "<｜tool▁calls▁begin｜>\(deepseek)<｜tool▁calls▁end｜>"),
        ]
        for (format, text) in cases {
            for offset in 0 ... text.count {
                let split = text.index(text.startIndex, offsetBy: offset)
                let processor = ToolCallProcessor(format: format, tools: tools)
                _ = processor.processChunk(String(text[..<split]))
                _ = processor.processChunk(String(text[split...]))
                processor.processEOS()
                #expect(processor.toolCalls.map(\.function.name) == ["first"])
                #expect(processor.rejectedToolCalls.isEmpty)
            }
        }
    }

    @Test("A section emits every call in source order")
    func batchedCalls() {
        for (format, text): (ToolCallFormat, String) in [
            (
                .kimiK2,
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.first:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_call_begin|>functions.second:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"
            ),
            (
                .deepseek,
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>first\n{}<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>second\n{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
            ),
        ] {
            let processor = ToolCallProcessor(format: format, tools: tools)
            _ = processor.processChunk(text)
            #expect(processor.toolCalls.map(\.function.name) == ["first", "second"])
        }
    }

    @Test("Native markers inside JSON strings remain argument data")
    func literalMarkers() throws {
        let cases: [(ToolCallFormat, String, String)] = [
            (
                .kimiK2, "<|tool_call_end|><|tool_calls_section_end|>",
                "<|tool_call_begin|>functions.first:0<|tool_call_argument_begin|>"
            ),
            (
                .deepseek, "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                "<｜tool▁call▁begin｜>function<｜tool▁sep｜>first\n"
            ),
        ]
        for (format, marker, prefix) in cases {
            let close = format == .kimiK2 ? "<|tool_call_end|>" : "<｜tool▁call▁end｜>"
            let text = prefix + "{\"value\":\"\(marker)\"}" + close
            let processor = ToolCallProcessor(format: format, tools: tools)
            for character in text { _ = processor.processChunk(String(character)) }
            let call = try #require(processor.toolCalls.first)
            #expect(call.function.arguments["value"] == .string(marker))
            #expect(processor.toolCalls.count == 1)
        }
    }

    @Test("Malformed JSON never becomes a DeepSeek call with string arguments")
    func malformedDeepSeek() {
        let processor = ToolCallProcessor(format: .deepseek, tools: tools)
        _ = processor.processChunk(
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>first\n{\"value\":}<｜tool▁call▁end｜>")
        processor.processEOS()
        #expect(processor.toolCalls.isEmpty)
        #expect(!processor.rejectedToolCalls.isEmpty)
    }

    @Test("DeepSeek dialect selection follows checkpoint metadata")
    func formatInference() {
        #expect(ToolCallFormat(toolParserType: "deepseek_v3") == .deepseek)
        #expect(ToolCallFormat.inferred(fromChatTemplate: "<｜tool▁calls▁begin｜>") == .deepseek)
        #expect(ToolCallFormat.infer(from: "gpt_oss") == nil)
    }
}
