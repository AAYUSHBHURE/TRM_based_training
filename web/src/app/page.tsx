import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {/* Hero Section */}
      <div className="container mx-auto px-6 py-20">
        <div className="text-center mb-16">
          <h1 className="text-6xl font-bold mb-4">
            <span className="bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
              TRM
            </span>
          </h1>
          <p className="text-2xl text-gray-300 mb-2">
            Tiny Recursive Model
          </p>
          <p className="text-lg text-gray-500 max-w-2xl mx-auto">
            An AI-powered educational gaming platform that teaches STEM reasoning
            through interactive puzzles with recursive visualization
          </p>
        </div>

        {/* Game Cards */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-6xl mx-auto">
          {/* Sudoku Forge */}
          <Link href="/sudoku" className="group">
            <div className="relative p-6 bg-gray-800/50 backdrop-blur rounded-2xl border border-gray-700 hover:border-blue-500 transition-all duration-300 hover:-translate-y-2 hover:shadow-xl hover:shadow-blue-500/20">
              <div className="text-5xl mb-4">🔢</div>
              <h3 className="text-2xl font-bold text-white mb-2 group-hover:text-blue-400 transition-colors">
                Sudoku Forge
              </h3>
              <p className="text-gray-400 mb-4">
                Math Logic • Ages 8-14
              </p>
              <p className="text-sm text-gray-500">
                Watch constraint propagation through z-trace mind webs.
                +40% deduction skills.
              </p>
              <div className="mt-4 flex items-center text-blue-400 text-sm font-medium">
                Play Now
                <svg className="w-4 h-4 ml-2 group-hover:translate-x-2 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </div>
          </Link>

          {/* Maze Navigator */}
          <Link href="/maze" className="group">
            <div className="relative p-6 bg-gray-800/50 backdrop-blur rounded-2xl border border-gray-700 hover:border-green-500 transition-all duration-300 hover:-translate-y-2 hover:shadow-xl hover:shadow-green-500/20">
              <div className="text-5xl mb-4">🌀</div>
              <h3 className="text-2xl font-bold text-white mb-2 group-hover:text-green-400 transition-colors">
                Maze Navigator
              </h3>
              <p className="text-gray-400 mb-4">
                Algorithms • Ages 10-16
              </p>
              <p className="text-sm text-gray-500">
                Visualize DFS/BFS with adaptive dead-ends that teach decomposition.
              </p>
              <div className="mt-4 flex items-center text-green-400 text-sm font-medium">
                Play Now
                <svg className="w-4 h-4 ml-2 group-hover:translate-x-2 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </div>
          </Link>

          {/* 2048 Challenge */}
          <Link href="/2048" className="group">
            <div className="relative p-6 bg-gray-800/50 backdrop-blur rounded-2xl border border-gray-700 hover:border-orange-500 transition-all duration-300 hover:-translate-y-2 hover:shadow-xl hover:shadow-orange-500/20">
              <div className="text-5xl mb-4">🎮</div>
              <h3 className="text-2xl font-bold text-white mb-2 group-hover:text-orange-400 transition-colors">
                2048 Challenge
              </h3>
              <p className="text-gray-400 mb-4">
                Strategy • Ages 10+
              </p>
              <p className="text-sm text-gray-500">
                Play manually or watch TRM use greedy heuristics to reach the 2048 tile.
              </p>
              <div className="mt-4 flex items-center text-orange-400 text-sm font-medium">
                Play Now
                <svg className="w-4 h-4 ml-2 group-hover:translate-x-2 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </div>
          </Link>

          {/* ARC Abstracter */}
          <div className="group opacity-60">
            <div className="relative p-6 bg-gray-800/50 backdrop-blur rounded-2xl border border-gray-700">
              <div className="absolute top-4 right-4 text-xs bg-purple-500/20 text-purple-300 px-2 py-1 rounded">
                Coming Soon
              </div>
              <div className="text-5xl mb-4">🔷</div>
              <h3 className="text-2xl font-bold text-white mb-2">
                ARC Abstracter
              </h3>
              <p className="text-gray-400 mb-4">
                Cognition • Ages 12+
              </p>
              <p className="text-sm text-gray-500">
                Pattern recognition with multi-abstraction z-traces.
                +30% transfer learning.
              </p>
            </div>
          </div>
        </div>

        {/* Features */}
        <div className="mt-20 text-center">
          <h2 className="text-3xl font-bold text-white mb-8">How TRM Works</h2>
          <div className="grid md:grid-cols-4 gap-6 max-w-4xl mx-auto">
            {[
              { icon: "📥", title: "Input (x)", desc: "Puzzle grid" },
              { icon: "🔄", title: "Recursion", desc: "T cycles of refinement" },
              { icon: "🧠", title: "Latent (z)", desc: "Reasoning traces" },
              { icon: "✨", title: "Output (ŷ)", desc: "Solution emerges" },
            ].map((step, idx) => (
              <div key={idx} className="p-4 bg-gray-800/30 rounded-xl">
                <div className="text-3xl mb-2">{step.icon}</div>
                <div className="font-bold text-white">{step.title}</div>
                <div className="text-sm text-gray-500">{step.desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Stats */}
        <div className="mt-16 flex justify-center gap-12">
          <div className="text-center">
            <div className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              5M
            </div>
            <div className="text-gray-500">Parameters</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold bg-gradient-to-r from-green-400 to-cyan-500 bg-clip-text text-transparent">
              87%
            </div>
            <div className="text-gray-500">Accuracy Target</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold bg-gradient-to-r from-yellow-400 to-orange-500 bg-clip-text text-transparent">
              50ms
            </div>
            <div className="text-gray-500">Inference</div>
          </div>
        </div>
      </div>
    </main>
  );
}
