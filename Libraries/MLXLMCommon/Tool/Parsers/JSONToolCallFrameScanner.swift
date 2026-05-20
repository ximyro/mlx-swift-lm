// Copyright © 2026 Apple Inc.

import Foundation

/// Locates Kimi and DeepSeek call boundaries without interpreting markers inside JSON strings.
struct JSONToolCallFrameScanner: Sendable {
    let sections: [(start: String, end: String)]
    let callStart: String
    let callEnd: String
    var allowsJSONFence = false
    var allowsUnwrappedCall = false

    func scan(_ text: String) -> (bodies: [Substring], end: String.Index)? {
        let section = sections.first { text.hasPrefix($0.start) }
        var remaining = text[...]
        if let section { remaining = remaining.dropFirst(section.start.count).trimmingWhitespace() }
        if allowsUnwrappedCall, let section, !remaining.hasPrefix(callStart),
            let brace = remaining.firstIndex(of: "{"),
            let json = JSONLeadingObjectScanner(startCharacter: "{").splitLeadingObject(
                from: String(remaining[brace...]))
        {
            let afterJSON = remaining.index(brace, offsetBy: json.object.count)
            let suffix = remaining[afterJSON...].trimmingWhitespace()
            guard suffix.hasPrefix(section.end) else { return nil }
            return (
                [remaining[..<afterJSON]],
                suffix.index(suffix.startIndex, offsetBy: section.end.count)
            )
        }
        var bodies: [Substring] = []
        repeat {
            guard remaining.hasPrefix(callStart) else { return nil }
            let body = remaining.dropFirst(callStart.count)
            guard let brace = body.firstIndex(of: "{"),
                let json = JSONLeadingObjectScanner(startCharacter: "{").splitLeadingObject(
                    from: String(body[brace...]))
            else { return nil }
            var suffix = body[body.index(brace, offsetBy: json.object.count)...]
                .trimmingWhitespace()
            if allowsJSONFence, suffix.hasPrefix("```") {
                suffix = suffix.dropFirst(3).trimmingWhitespace()
            }
            guard suffix.hasPrefix(callEnd) else { return nil }
            bodies.append(body[..<suffix.startIndex])
            let end = suffix.index(suffix.startIndex, offsetBy: callEnd.count)
            guard let section else { return (bodies, end) }
            remaining = text[end...].trimmingWhitespace()
            if remaining.hasPrefix(section.end) {
                return (bodies, remaining.index(remaining.startIndex, offsetBy: section.end.count))
            }
        } while !remaining.isEmpty
        return nil
    }
}
