"use client";

import React, { useState, useCallback, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ── Tile styles ───────────────────────────────────────────────────────────────
function getTileStyle(value: number): string {
    const map: Record<number, string> = {
        0: "bg-gray-700/40 text-transparent",
        2: "bg-amber-100 text-gray-800",
        4: "bg-amber-200 text-gray-800",
        8: "bg-orange-400 text-white",
        16: "bg-orange-500 text-white",
        32: "bg-orange-600 text-white",
        64: "bg-red-500 text-white",
        128: "bg-yellow-400 text-white",
        256: "bg-yellow-500 text-white",
        512: "bg-yellow-600 text-white",
        1024: "bg-orange-500 text-white",
        2048: "bg-gradient-to-br from-yellow-400 to-orange-500 text-white shadow-lg shadow-orange-500/50",
    };
    return map[value] ?? "bg-orange-700 text-white";
}

function getTileTextSize(value: number): string {
    if (value >= 1024) return "text-lg";
    if (value >= 128) return "text-xl";
    return "text-2xl";
}

// ── Game logic ────────────────────────────────────────────────────────────────
function mergeRow(row: number[]): [number[], number] {
    const tiles = row.filter(x => x !== 0);
    let score = 0;
    const merged: number[] = [];
    let i = 0;
    while (i < tiles.length) {
        if (i + 1 < tiles.length && tiles[i] === tiles[i + 1]) {
            merged.push(tiles[i] * 2);
            score += tiles[i] * 2;
            i += 2;
        } else {
            merged.push(tiles[i]);
            i++;
        }
    }
    while (merged.length < 4) merged.push(0);
    return [merged, score];
}

function applyMove(board: number[][], dir: string): [number[][], number] {
    const b = board.map(r => [...r]);
    let score = 0;
    if (dir === "left") {
        for (let r = 0; r < 4; r++) {
            const [row, s] = mergeRow(b[r]);
            b[r] = row; score += s;
        }
    } else if (dir === "right") {
        for (let r = 0; r < 4; r++) {
            const [row, s] = mergeRow([...b[r]].reverse());
            b[r] = row.reverse(); score += s;
        }
    } else if (dir === "up") {
        for (let c = 0; c < 4; c++) {
            const col = [b[0][c], b[1][c], b[2][c], b[3][c]];
            const [merged, s] = mergeRow(col);
            score += s;
            for (let r = 0; r < 4; r++) b[r][c] = merged[r];
        }
    } else if (dir === "down") {
        for (let c = 0; c < 4; c++) {
            const col = [b[0][c], b[1][c], b[2][c], b[3][c]];
            const [merged, s] = mergeRow([...col].reverse());
            score += s;
            const rev = [...merged].reverse();
            for (let r = 0; r < 4; r++) b[r][c] = rev[r];
        }
    }
    return [b, score];
}

function boardChanged(b1: number[][], b2: number[][]): boolean {
    return b1.some((row, r) => row.some((v, c) => v !== b2[r][c]));
}

function addTile(board: number[][]): number[][] {
    const empties: [number, number][] = [];
    for (let r = 0; r < 4; r++)
        for (let c = 0; c < 4; c++)
            if (board[r][c] === 0) empties.push([r, c]);
    if (empties.length === 0) return board;
    const [r, c] = empties[Math.floor(Math.random() * empties.length)];
    const nb = board.map(row => [...row]);
    nb[r][c] = Math.random() < 0.1 ? 4 : 2;
    return nb;
}

function newBoard(): number[][] {
    let b: number[][] = Array.from({ length: 4 }, () => new Array(4).fill(0));
    b = addTile(b);
    b = addTile(b);
    return b;
}

function isGameOver(board: number[][]): boolean {
    for (let r = 0; r < 4; r++)
        for (let c = 0; c < 4; c++) {
            if (board[r][c] === 0) return false;
            if (c + 1 < 4 && board[r][c] === board[r][c + 1]) return false;
            if (r + 1 < 4 && board[r][c] === board[r + 1][c]) return false;
        }
    return true;
}

function hasWon(board: number[][]): boolean {
    return board.some(row => row.some(v => v >= 2048));
}

function boardHeuristic(board: number[][]): number {
    let empties = 0, mono = 0, maxTile = 0;
    for (let r = 0; r < 4; r++)
        for (let c = 0; c < 4; c++) {
            if (board[r][c] === 0) empties++;
            if (board[r][c] > maxTile) maxTile = board[r][c];
        }
    const corners = [board[0][0], board[0][3], board[3][0], board[3][3]];
    const cornerBonus = corners.includes(maxTile) ? maxTile * 3 : 0;
    for (let r = 0; r < 4; r++)
        for (let c = 0; c < 3; c++)
            if (board[r][c] >= board[r][c + 1]) mono += board[r][c + 1];
    for (let c = 0; c < 4; c++)
        for (let r = 0; r < 3; r++)
            if (board[r][c] >= board[r + 1][c]) mono += board[r + 1][c];
    return empties * 150 + cornerBonus + mono;
}

function greedyMove(board: number[][]): string | null {
    let bestScore = -Infinity, bestDir: string | null = null;
    for (const d of ["up", "left", "right", "down"]) {
        const [nb] = applyMove(board, d);
        if (boardChanged(board, nb)) {
            const s = boardHeuristic(nb);
            if (s > bestScore) { bestScore = s; bestDir = d; }
        }
    }
    return bestDir;
}

const DIR_ICONS: Record<string, string> = { up: "↑", left: "←", right: "→", down: "↓" };

// ── Component ─────────────────────────────────────────────────────────────────
export function Game2048() {
    const [board, setBoard] = useState<number[][]>(newBoard);
    const [score, setScore] = useState(0);
    const [bestScore, setBestScore] = useState(0);
    const [gameOver, setGameOver] = useState(false);
    const [won, setWon] = useState(false);
    const [isAiRunning, setIsAiRunning] = useState(false);
    const [recursionStep, setRecursionStep] = useState(0);
    const [thinking, setThinking] = useState<string | null>(null);
    const aiRef = useRef(false);

    const handleMove = useCallback((dir: string) => {
        setBoard(prev => {
            const [nb, s] = applyMove(prev, dir);
            if (!boardChanged(prev, nb)) return prev;
            const withTile = addTile(nb);
            setScore(sc => {
                const next = sc + s;
                setBestScore(best => Math.max(best, next));
                return next;
            });
            if (hasWon(withTile)) setWon(true);
            if (isGameOver(withTile)) setGameOver(true);
            return withTile;
        });
    }, []);

    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (isAiRunning || gameOver) return;
            const map: Record<string, string> = {
                ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
            };
            if (map[e.key]) { e.preventDefault(); handleMove(map[e.key]); }
        };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, [handleMove, isAiRunning, gameOver]);

    const runAI = useCallback(async () => {
        setIsAiRunning(true);
        aiRef.current = true;

        let currentBoard = board;
        let steps = 0;
        const maxSteps = 200;

        while (aiRef.current && !isGameOver(currentBoard) && !hasWon(currentBoard) && steps < maxSteps) {
            // Simulate TRM recursion cycles
            for (let cycle = 1; cycle <= 3; cycle++) {
                if (!aiRef.current) break;
                setRecursionStep(cycle);
                setThinking(greedyMove(currentBoard));
                await new Promise(r => setTimeout(r, 120));
            }

            const dir = greedyMove(currentBoard);
            if (!dir) break;

            const [nb, s] = applyMove(currentBoard, dir);
            if (!boardChanged(currentBoard, nb)) break;
            const withTile = addTile(nb);

            currentBoard = withTile;
            setBoard(withTile);
            setScore(sc => {
                const next = sc + s;
                setBestScore(best => Math.max(best, next));
                return next;
            });

            if (hasWon(withTile)) { setWon(true); break; }
            if (isGameOver(withTile)) { setGameOver(true); break; }

            steps++;
            await new Promise(r => setTimeout(r, 80));
        }

        aiRef.current = false;
        setIsAiRunning(false);
        setRecursionStep(0);
        setThinking(null);
    }, [board]);

    const stopAI = () => { aiRef.current = false; };

    const resetGame = () => {
        aiRef.current = false;
        setBoard(newBoard());
        setScore(0);
        setGameOver(false);
        setWon(false);
        setIsAiRunning(false);
        setRecursionStep(0);
        setThinking(null);
    };

    const maxTile = Math.max(...board.flat());

    return (
        <div className="flex flex-col items-center gap-8 p-6">
            {/* Header */}
            <div className="text-center">
                <h2 className="text-3xl font-bold bg-gradient-to-r from-orange-400 to-yellow-500 bg-clip-text text-transparent">
                    2048 Challenge
                </h2>
                <p className="text-gray-400 mt-2">
                    Play manually or watch TRM reason its way to 2048
                </p>
            </div>

            {/* Score Bar */}
            <div className="flex gap-6">
                <div className="px-6 py-3 bg-orange-900/40 rounded-xl text-center border border-orange-800/50">
                    <div className="text-xs text-orange-400 font-medium uppercase tracking-wide">Score</div>
                    <motion.div
                        key={score}
                        initial={{ scale: 1.2 }}
                        animate={{ scale: 1 }}
                        className="text-2xl font-bold text-white"
                    >
                        {score}
                    </motion.div>
                </div>
                <div className="px-6 py-3 bg-gray-800 rounded-xl text-center border border-gray-700">
                    <div className="text-xs text-gray-400 font-medium uppercase tracking-wide">Best</div>
                    <div className="text-2xl font-bold text-yellow-400">{bestScore}</div>
                </div>
                <div className="px-6 py-3 bg-gray-800 rounded-xl text-center border border-gray-700">
                    <div className="text-xs text-gray-400 font-medium uppercase tracking-wide">Max Tile</div>
                    <div className="text-2xl font-bold text-orange-400">{maxTile}</div>
                </div>
            </div>

            {/* TRM Recursion Status */}
            <AnimatePresence>
                {isAiRunning && (
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="flex items-center gap-4 px-6 py-3 bg-orange-900/40 rounded-full border border-orange-800/50"
                    >
                        <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                            className="w-5 h-5 border-2 border-orange-400 border-t-transparent rounded-full"
                        />
                        <span className="text-orange-300 font-medium">
                            TRM Cycle {recursionStep}/3
                        </span>
                        {thinking && (
                            <span className="text-yellow-400 font-bold text-xl w-6 text-center">
                                {DIR_ICONS[thinking]}
                            </span>
                        )}
                        <div className="flex gap-1">
                            {[1, 2, 3].map(c => (
                                <motion.div
                                    key={c}
                                    className={`w-3 h-3 rounded-full ${c <= recursionStep ? "bg-orange-400" : "bg-gray-600"}`}
                                    animate={c === recursionStep ? { scale: [1, 1.3, 1] } : {}}
                                    transition={{ repeat: Infinity, duration: 0.5 }}
                                />
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Game Board */}
            <motion.div
                className="relative p-3 bg-gray-900 rounded-2xl shadow-2xl border border-gray-700"
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
            >
                {/* Win / Game Over overlay */}
                <AnimatePresence>
                    {(gameOver || won) && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="absolute inset-0 flex flex-col items-center justify-center rounded-2xl z-10 bg-gray-900/85 backdrop-blur-sm"
                        >
                            <div className={`text-4xl font-bold mb-2 ${won ? "text-yellow-400" : "text-red-400"}`}>
                                {won ? "You Won!" : "Game Over"}
                            </div>
                            <div className="text-gray-300 text-xl mb-4">Score: {score}</div>
                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={resetGame}
                                className="px-6 py-3 rounded-lg font-bold bg-gradient-to-r from-orange-500 to-yellow-500 text-white"
                            >
                                New Game
                            </motion.button>
                        </motion.div>
                    )}
                </AnimatePresence>

                <div className="grid grid-cols-4 gap-2">
                    {board.flat().map((value, idx) => (
                        <motion.div
                            key={idx}
                            animate={{ scale: value > 0 ? 1 : 0.95 }}
                            transition={{ type: "spring", damping: 20 }}
                            className={`
                                w-16 h-16 flex items-center justify-center
                                rounded-lg font-bold
                                ${getTileTextSize(value)}
                                ${getTileStyle(value)}
                            `}
                        >
                            {value > 0 ? value : ""}
                        </motion.div>
                    ))}
                </div>
            </motion.div>

            {/* Controls */}
            <div className="flex gap-4 flex-wrap justify-center">
                {!isAiRunning ? (
                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={runAI}
                        disabled={gameOver || won}
                        className={`
                            px-8 py-3 rounded-lg font-bold text-lg
                            ${gameOver || won
                                ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                                : "bg-gradient-to-r from-orange-500 to-yellow-500 text-white hover:from-orange-600 hover:to-yellow-600"
                            }
                            transition-all shadow-lg
                        `}
                    >
                        TRM Auto-Play
                    </motion.button>
                ) : (
                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={stopAI}
                        className="px-8 py-3 rounded-lg font-bold text-lg bg-red-700 text-white hover:bg-red-600 transition-all shadow-lg"
                    >
                        Stop AI
                    </motion.button>
                )}
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={resetGame}
                    className="px-6 py-3 rounded-lg font-medium bg-gray-700 text-gray-300 hover:bg-gray-600 transition-all"
                >
                    New Game
                </motion.button>
            </div>

            {/* Keyboard hint */}
            {!isAiRunning && !gameOver && !won && (
                <p className="text-sm text-gray-500">
                    Use arrow keys to play manually
                </p>
            )}

            {/* TRM info */}
            {!isAiRunning && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex gap-8 p-4 bg-gray-800/50 rounded-lg border border-gray-700 max-w-sm w-full"
                >
                    <div className="text-center flex-1">
                        <div className="text-xl font-bold text-orange-400">Greedy</div>
                        <div className="text-xs text-gray-400">AI Strategy</div>
                    </div>
                    <div className="text-center flex-1">
                        <div className="text-xl font-bold text-yellow-400">3</div>
                        <div className="text-xs text-gray-400">Recursion Cycles</div>
                    </div>
                    <div className="text-center flex-1">
                        <div className="text-xl font-bold text-amber-400">Corner</div>
                        <div className="text-xs text-gray-400">Heuristic</div>
                    </div>
                </motion.div>
            )}
        </div>
    );
}

export default Game2048;
