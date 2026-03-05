import SudokuForge from "@/components/SudokuForge";
import Link from "next/link";

export default function SudokuPage() {
    return (
        <main className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
            {/* Navigation */}
            <nav className="border-b border-gray-800 bg-gray-900/50 backdrop-blur sticky top-0 z-50">
                <div className="container mx-auto px-6 py-4 flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-2 text-xl font-bold">
                        <span className="bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                            TRM
                        </span>
                    </Link>
                    <div className="flex items-center gap-6">
                        <Link href="/sudoku" className="text-blue-400 font-medium">
                            Sudoku
                        </Link>
                        <Link href="/maze" className="text-gray-400 hover:text-white transition-colors">
                            Maze
                        </Link>
                    </div>
                </div>
            </nav>

            {/* Content */}
            <div className="container mx-auto py-8">
                <SudokuForge />
            </div>
        </main>
    );
}
