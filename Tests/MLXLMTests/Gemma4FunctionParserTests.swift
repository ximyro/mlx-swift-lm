import Foundation
import MLXLMCommon
import Testing

struct Gemma4FunctionParserTests {
    @Test("Gemma4 parser handles scalar argument types")
    func testScalarArguments() throws {
        let parser = Gemma4FunctionParser()
        let content = #"<|tool_call>call:search{query:<|"|>hello<|"|>,limit:10,enabled:true,missing:null}<tool_call|>"#

        let call = try #require(parser.parse(content: content, tools: nil))

        #expect(call.function.name == "search")
        #expect(call.function.arguments["query"] == .string("hello"))
        #expect(call.function.arguments["limit"] == .int(10))
        #expect(call.function.arguments["enabled"] == .bool(true))
        #expect(call.function.arguments["missing"] == .null)
    }

    @Test("Gemma4 parser handles nested objects and arrays")
    func testNestedArguments() throws {
        let parser = Gemma4FunctionParser()
        let content = #"<|tool_call>call:configure{settings:{enabled:true,tags:[<|"|>a<|"|>,<|"|>b<|"|>]}}<tool_call|>"#

        let call = try #require(parser.parse(content: content, tools: nil))

        #expect(
            call.function.arguments["settings"]
                == .object([
                    "enabled": .bool(true),
                    "tags": .array([.string("a"), .string("b")]),
                ]))
    }

    @Test("Gemma4 parser returns multiple calls from one block")
    func testMultipleCalls() throws {
        let parser = Gemma4FunctionParser()
        let content = #"<|tool_call>call:read{path:<|"|>/a.go<|"|>}call:read{path:<|"|>/b.go<|"|>}<tool_call|>"#

        let calls = parser.parseEOS(content, tools: nil)

        #expect(calls.count == 2)
        try #require(calls.count == 2)
        #expect(calls[0].function.name == "read")
        #expect(calls[0].function.arguments["path"] == .string("/a.go"))
        #expect(calls[1].function.arguments["path"] == .string("/b.go"))
    }

    @Test("Gemma4 parser ignores braces and colons inside string values")
    func testBracesAndColonsInsideString() throws {
        let parser = Gemma4FunctionParser()
        let content = #"<|tool_call>call:write{code:<|"|>if (x) { return "host:8080"; }<|"|>}<tool_call|>"#

        let call = try #require(parser.parse(content: content, tools: nil))

        #expect(call.function.arguments["code"] == .string(#"if (x) { return "host:8080"; }"#))
    }

    @Test("Gemma4 parser quotes bare string values")
    func testBareStringValues() throws {
        let parser = Gemma4FunctionParser()
        let content = "<|tool_call>call:set_state{domain:light,tags:[alpha,beta],ready:true}<tool_call|>"

        let call = try #require(parser.parse(content: content, tools: nil))

        #expect(call.function.arguments["domain"] == .string("light"))
        #expect(call.function.arguments["tags"] == .array([.string("alpha"), .string("beta")]))
        #expect(call.function.arguments["ready"] == .bool(true))
    }

    @Test("Gemma4 parser accepts missing end delimiter")
    func testMissingEndDelimiter() throws {
        let parser = Gemma4FunctionParser()
        let content = #"<|tool_call>call:read{path:<|"|>/tmp/file.go<|"|>}"#

        let call = try #require(parser.parse(content: content, tools: nil))

        #expect(call.function.name == "read")
        #expect(call.function.arguments["path"] == .string("/tmp/file.go"))
    }

    @Test("Gemma4 parser accepts hyphenated function names")
    func testHyphenatedFunctionName() throws {
        let parser = Gemma4FunctionParser()
        let content = #"<|tool_call>call:get-weather{city:<|"|>Tokyo<|"|>}<tool_call|>"#

        let call = try #require(parser.parse(content: content, tools: nil))

        #expect(call.function.name == "get-weather")
        #expect(call.function.arguments["city"] == .string("Tokyo"))
    }

    @Test("Gemma4 parser strips think tags before tool calls")
    func testThinkTagsStripped() throws {
        let parser = Gemma4FunctionParser()
        let content = #"<think>Need a file.</think><|tool_call>call:read{path:<|"|>/tmp/file.go<|"|>}<tool_call|>"#

        let call = try #require(parser.parse(content: content, tools: nil))

        #expect(call.function.name == "read")
        #expect(call.function.arguments["path"] == .string("/tmp/file.go"))
    }
}
