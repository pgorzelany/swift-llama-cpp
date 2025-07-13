import Foundation

extension Task {
    public func cancelAndWait() async {
        self.cancel()
        _ = try? await self.value
    }
}
