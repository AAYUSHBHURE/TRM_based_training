"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useTRMStore } from "@/lib/store";

// Sample puzzle
const SAMPLE_PUZZLE = [
    5, 3, 0, 0, 7, 0, 0, 0, 0,
    6, 0, 0, 1, 9, 5, 0, 0, 0,
    0, 9, 8, 0, 0, 0, 0, 6, 0,
    8, 0, 0, 0, 6, 0, 0, 0, 3,
    4, 0, 0, 8, 0, 3, 0, 0, 1,
    7, 0, 0, 0, 2, 0, 0, 0, 6,
    0, 6, 0, 0, 0, 0, 2, 8, 0,
    0, 0, 0, 4, 1, 9, 0, 0, 5,
    0, 0, 0, 0, 8, 0, 0, 7, 9,
];

const SAMPLE_SOLUTION = [
    5, 3, 4, 6, 7, 8, 9, 1, 2,
    6, 7, 2, 1, 9, 5, 3, 4, 8,
    1, 9, 8, 3, 4, 2, 5, 6, 7,
    8, 5, 9, 7, 6, 1, 4, 2, 3,
    4, 2, 6, 8, 5, 3, 7, 9, 1,
    7, 1, 3, 9, 2, 4, 8, 5, 6,
    9, 6, 1, 5, 3, 7, 2, 8, 4,
    2, 8, 7, 4, 1, 9, 6, 3, 5,
    3, 4, 5, 2, 8, 6, 1, 7, 9,
];

interface CellProps {
    value: number;
    isGiven: boolean;
    isSolved: boolean;
    isHighlighted: boolean;
    row: number;
    col: number;
}

function Cell({ value, isGiven, isSolved, isHighlighted, row, col }: CellProps) {
    const borderClasses = [
        col % 3 === 0 && col > 0 ? "border-l-2 border-l-blue-400" : "",
        row % 3 === 0 && row > 0 ? "border-t-2 border-t-blue-400" : "",
    ].join(" ");

    return (
        <motion.div
            initial={isSolved ? { scale: 0.5, opacity: 0 } : false}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", damping: 20, delay: isSolved ? (row * 9 + col) * 0.01 : 0 }}
            className={`
        w-12 h-12 flex items-center justify-center
        border border-gray-600
        ${borderClasses}
        ${isGiven ? "bg-gray-700 text-white" : "bg-gray-800"}
        ${isSolved && !isGiven ? "bg-green-900 text-green-300" : ""}
        ${isHighlighted ? "ring-2 ring-yellow-400" : ""}
        font-bold text-xl
        transition-all duration-200
        hover:bg-gray-600 cursor-pointer
      `}
        >
            {value !== 0 ? value : ""}
        </motion.div>
    );
}

export function SudokuForge() {
    const { sudoku, setPuzzle, setPrediction, setLoading } = useTRMStore();
    const [currentPuzzle, setCurrentPuzzle] = useState<number[]>(SAMPLE_PUZZLE);
    const [currentSolution, setCurrentSolution] = useState<number[]>([]);
    const [highlightedCell, setHighlightedCell] = useState<number | null>(null);
    const [recursionStep, setRecursionStep] = useState(0);
    const [isRunning, setIsRunning] = useState(false);

    const solvePuzzle = useCallback(async () => {
        setIsRunning(true);
        setRecursionStep(0);

        // Simulate TRM recursion cycles
        for (let cycle = 1; cycle <= 3; cycle++) {
            setRecursionStep(cycle);

            // Simulate latent updates
            for (let i = 0; i < 6; i++) {
                await new Promise((r) => setTimeout(r, 50));
            }

            // Progressive solve for visual effect
            const progress = cycle / 3;
            const partialSolution = currentPuzzle.map((v, idx) => {
                if (v !== 0) return v;
                if (Math.random() < progress) return SAMPLE_SOLUTION[idx];
                return 0;
            });
            setCurrentSolution(partialSolution);

            await new Promise((r) => setTimeout(r, 200));
        }

        // Final solution
        setCurrentSolution(SAMPLE_SOLUTION);
        setIsRunning(false);
    }, [currentPuzzle]);

    const resetPuzzle = () => {
        setCurrentSolution([]);
        setRecursionStep(0);
    };

    const displayGrid = currentSolution.length > 0 ? currentSolution : currentPuzzle;

    return (
        <div className="flex flex-col items-center gap-8 p-6">
            {/* Header */}
            <div className="text-center">
                <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                    Sudoku Forge
                </h2>
                <p className="text-gray-400 mt-2">
                    Watch TRM solve puzzles through recursive reasoning
                </p>
            </div>

            {/* Recursion Status */}
            <AnimatePresence>
                {isRunning && (
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="flex items-center gap-4 px-6 py-3 bg-blue-900/50 rounded-full"
                    >
                        <div className="relative w-6 h-6">
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                                className="absolute inset-0 border-2 border-blue-400 border-t-transparent rounded-full"
                            />
                        </div>
                        <span className="text-blue-300 font-medium">
                            Recursion Cycle {recursionStep}/3
                        </span>
                        <div className="flex gap-1">
                            {[1, 2, 3].map((c) => (
                                <motion.div
                                    key={c}
                                    className={`w-3 h-3 rounded-full ${c <= recursionStep ? "bg-green-400" : "bg-gray-600"
                                        }`}
                                    animate={c === recursionStep ? { scale: [1, 1.3, 1] } : {}}
                                    transition={{ repeat: Infinity, duration: 0.5 }}
                                />
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Grid */}
            <motion.div
                className="grid grid-cols-9 gap-0 p-4 bg-gray-900 rounded-xl shadow-2xl border border-gray-700"
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
            >
                {displayGrid.map((value, idx) => {
                    const row = Math.floor(idx / 9);
                    const col = idx % 9;
                    const isGiven = currentPuzzle[idx] !== 0;
                    const isSolved = currentSolution.length > 0 && currentSolution[idx] !== 0 && !isGiven;

                    return (
                        <Cell
                            key={idx}
                            value={value}
                            isGiven={isGiven}
                            isSolved={isSolved}
                            isHighlighted={highlightedCell === idx}
                            row={row}
                            col={col}
                        />
                    );
                })}
            </motion.div>

            {/* Controls */}
            <div className="flex gap-4">
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={solvePuzzle}
                    disabled={isRunning}
                    className={`
            px-8 py-3 rounded-lg font-bold text-lg
            ${isRunning
                            ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                            : "bg-gradient-to-r from-blue-500 to-purple-500 text-white hover:from-blue-600 hover:to-purple-600"
                        }
            transition-all shadow-lg
          `}
                >
                    {isRunning ? "Solving..." : "🧠 Solve with TRM"}
                </motion.button>

                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={resetPuzzle}
                    className="px-6 py-3 rounded-lg font-medium bg-gray-700 text-gray-300 hover:bg-gray-600 transition-all"
                >
                    Reset
                </motion.button>
            </div>

            {/* Stats */}
            {currentSolution.length > 0 && !isRunning && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex gap-8 p-4 bg-gray-800 rounded-lg"
                >
                    <div className="text-center">
                        <div className="text-2xl font-bold text-green-400">100%</div>
                        <div className="text-sm text-gray-400">Accuracy</div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold text-blue-400">3</div>
                        <div className="text-sm text-gray-400">Cycles</div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold text-purple-400">45ms</div>
                        <div className="text-sm text-gray-400">Inference</div>
                    </div>
                </motion.div>
            )}
        </div>
    );
}

export default SudokuForge;
