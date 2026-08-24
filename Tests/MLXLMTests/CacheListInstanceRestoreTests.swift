import MLX
import Testing

@testable import MLXLMCommon

extension MLXTestingSuite {
    struct CacheListInstanceRestoreTests {
        @Test("Instance restore rebuilds nested caches and leaves them intact on invalid metadata")
        func restoreNestedCaches() throws {
            let recurrent = MambaCache()
            recurrent[0] = MLXArray([Float(3), 4])
            recurrent[1] = MLXArray([Float(5), 6])
            let attention = KVCacheSimple()
            let keys = MLXArray.ones([1, 1, 3, 4])
            _ = attention.update(keys: keys, values: keys * 2)
            let source = CacheList(recurrent, CacheList(attention))
            let target = CacheList(KVCacheSimple())

            try target.restore(state: source.state, metaState: source.metaState)

            #expect(target.children.count == 2)
            #expect(target[0] is MambaCache)
            let nested = try #require(target[1] as? CacheList)
            #expect(nested[0].offset == 3)
            #expect(target.metaState == source.metaState)
            #expect(target.state.count == source.state.count)
            for (actual, expected) in zip(target.state, source.state) {
                #expect(allClose(actual, expected).item(Bool.self))
            }

            #expect(throws: (any Error).self) {
                try target.restore(state: [], metaState: ["2", "KVCacheSimple"])
            }
            #expect(target.metaState == source.metaState)
        }
    }
}
