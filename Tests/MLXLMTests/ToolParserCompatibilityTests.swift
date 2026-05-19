import Foundation
import MLXLMCommon
import Testing

struct ToolParserCompatibilityTests {
    @Test("Bracket calls recover only declared tools outside response data")
    func bracketRecovery() throws {
        let text = #"[Calling tool: read({"path":"literal )] stays data"})]"#
        let tools: [[String: any Sendable]] = [
            ["function": ["name": "read"] as [String: any Sendable]]
        ]
        for offset in 0 ... text.count {
            let split = text.index(text.startIndex, offsetBy: offset)
            let processor = ToolCallProcessor(format: .qwen35, tools: tools)
            _ = processor.processChunk(String(text[..<split]))
            _ = processor.processChunk(String(text[split...]))
            processor.processEOS()
            let call = try #require(processor.toolCalls.first)
            #expect(processor.toolCalls.count == 1)
            #expect(call.function.arguments["path"] == .string("literal )] stays data"))
            #expect(processor.recoveryEvents.first?.dialect == .qwenBracket)
        }
        for protected in ["<think>\(text)</think>", "`\(text)`", "```\n\(text)\n```"] {
            let processor = ToolCallProcessor(format: .qwen35, tools: tools)
            _ = processor.processChunk(protected)
            processor.processEOS()
            #expect(processor.toolCalls.isEmpty)
        }
        let undeclared = ToolCallProcessor(format: .qwen35, tools: [])
        _ = undeclared.processChunk(text)
        undeclared.processEOS()
        #expect(undeclared.toolCalls.isEmpty)
    }

    @Test("Qwen and Gemma batches retain call order at every streaming boundary")
    func batchStreamingBoundaries() {
        let cases: [(ToolCallFormat, String)] = [
            (
                .qwen35,
                "<tool_call><function=first></function><function=second></function></tool_call>"
            ),
            (.gemma4, "<|tool_call>call:first{}call:second{}<tool_call|>"),
        ]
        let tools: [[String: any Sendable]] = ["first", "second"].map {
            ["function": ["name": $0] as [String: any Sendable]]
        }
        for (format, text) in cases {
            for offset in 0 ... text.count {
                let split = text.index(text.startIndex, offsetBy: offset)
                let processor = ToolCallProcessor(format: format, tools: tools)
                _ = processor.processChunk(String(text[..<split]))
                _ = processor.processChunk(String(text[split...]))
                #expect(processor.toolCalls.map(\.function.name) == ["first", "second"])
                processor.processEOS()
                #expect(processor.toolCalls.count == 2)
                #expect(processor.rejectedToolCalls.isEmpty)
            }
        }
    }

    @Test("An incomplete second call never executes a partial batch")
    func incompleteBatch() {
        for (format, text): (ToolCallFormat, String) in [
            (.qwen35, "<tool_call><function=first></function><function=second><parameter=x>"),
            (.gemma4, "<|tool_call>call:first{}call:second{x:"),
        ] {
            let processor = ToolCallProcessor(format: format)
            _ = processor.processChunk(text)
            processor.processEOS()
            #expect(processor.toolCalls.isEmpty)
        }
    }

    @Test("Recovery cannot execute the first call of a malformed Gemma batch")
    func malformedBatchWithRecovery() {
        let tools: [[String: any Sendable]] = [
            ["function": ["name": "first"] as [String: any Sendable]]
        ]
        for format: ToolCallFormat in [.gemma4, .qwen35] {
            for suffix in ["call:second{x:", "garbage"] {
                let processor = ToolCallProcessor(format: format, tools: tools)
                _ = processor.processChunk("<|tool_call>call:first{}\(suffix)<tool_call|>")
                processor.processEOS()
                #expect(processor.toolCalls.isEmpty)
                #expect(!processor.rejectedToolCalls.isEmpty)
            }
        }
    }

    @Test("XML preserves strings under nullable and ambiguous parameter schemas")
    func declaredStringSchemas() throws {
        let schemas: [[String: any Sendable]] = [
            ["type": ["string", "null"]],
            ["anyOf": [["type": "string"], ["type": "null"]]],
            ["enum": ["42", "true"]],
            ["$ref": "#/$defs/text"],
            [:],
        ]
        for schema in schemas {
            let tools: [[String: any Sendable]] = [
                [
                    "function": [
                        "name": "configure",
                        "parameters": ["properties": ["value": schema]],
                    ] as [String: any Sendable]
                ]
            ]
            for value in ["42", "true"] {
                let text =
                    "<tool_call><function=configure><parameter=value>\(value)</parameter></function></tool_call>"
                let call = try #require(
                    ToolCallFormat.qwen35.createParser().parse(content: text, tools: tools))
                #expect(call.function.arguments["value"] == .string(value))
            }
        }
    }

    @Test("XML scalar inference preserves declared strings")
    func xmlScalarArguments() throws {
        let parser = ToolCallFormat.qwen35.createParser()
        let content =
            "<tool_call><function=configure><parameter=count>42</parameter><parameter=enabled>true</parameter></function></tool_call>"
        let inferred = try #require(parser.parse(content: content, tools: nil))
        #expect(inferred.function.arguments["count"] == .int(42))
        #expect(inferred.function.arguments["enabled"] == .bool(true))

        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "configure",
                    "parameters": ["properties": ["count": ["type": "string"]]],
                ] as [String: any Sendable]
            ]
        ]
        let declared = try #require(parser.parse(content: content, tools: tools))
        #expect(declared.function.arguments["count"] == .string("42"))
    }
}
