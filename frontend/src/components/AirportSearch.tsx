"use client";

import { useState, useRef, useEffect } from "react";

interface Airport {
    code: string;
    name: string;
}

interface Props {
    airports: Airport[];
    value: string;
    onChange: (code: string) => void;
    placeholder?: string;
    id?: string;
}

export default function AirportSearch({ airports, value, onChange, placeholder = "Type city or code", id }: Props) {
    const [query, setQuery] = useState("");
    const [open, setOpen] = useState(false);
    const [highlighted, setHighlighted] = useState(-1);
    const wrapRef = useRef<HTMLDivElement>(null);
    const listRef = useRef<HTMLUListElement>(null);

    // Sync display text when value changes from outside
    useEffect(() => {
        if (value) {
            const match = airports.find(a => a.code === value);
            setQuery(match ? match.name : value);
        } else {
            setQuery("");
        }
    }, [value, airports]);

    // Filter airports by query (search both code and name)
    const filtered = query.trim()
        ? airports.filter(a => {
            const q = query.toLowerCase();
            return a.code.toLowerCase().includes(q) || a.name.toLowerCase().includes(q);
        }).slice(0, 12)
        : airports.slice(0, 12);

    // Click outside to close
    useEffect(() => {
        const handler = (e: MouseEvent) => {
            if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
        };
        document.addEventListener("mousedown", handler);
        return () => document.removeEventListener("mousedown", handler);
    }, []);

    // Scroll highlighted item into view
    useEffect(() => {
        if (highlighted >= 0 && listRef.current) {
            const el = listRef.current.children[highlighted] as HTMLElement;
            el?.scrollIntoView({ block: "nearest" });
        }
    }, [highlighted]);

    const select = (a: Airport) => {
        onChange(a.code);
        setQuery(a.name);
        setOpen(false);
        setHighlighted(-1);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "ArrowDown") {
            e.preventDefault();
            setHighlighted(h => Math.min(h + 1, filtered.length - 1));
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setHighlighted(h => Math.max(h - 1, 0));
        } else if (e.key === "Enter" && highlighted >= 0) {
            e.preventDefault();
            select(filtered[highlighted]);
        } else if (e.key === "Escape") {
            setOpen(false);
        }
    };

    return (
        <div ref={wrapRef} style={{ position: "relative" }}>
            <input
                id={id}
                className="field-input"
                placeholder={placeholder}
                value={query}
                onChange={e => {
                    setQuery(e.target.value);
                    onChange(""); // reset code while typing
                    setOpen(true);
                    setHighlighted(-1);
                }}
                onFocus={() => setOpen(true)}
                onKeyDown={handleKeyDown}
                autoComplete="off"
            />
            {open && filtered.length > 0 && (
                <ul
                    ref={listRef}
                    style={{
                        position: "absolute", top: "100%", left: 0, right: 0,
                        zIndex: 50, maxHeight: 240, overflowY: "auto",
                        background: "var(--surface-2)", border: "1px solid var(--border)",
                        borderRadius: 10, marginTop: 4, padding: "4px 0",
                        boxShadow: "0 12px 32px rgba(0,0,0,0.5)",
                        listStyle: "none",
                    }}
                >
                    {filtered.map((a, i) => (
                        <li
                            key={a.code}
                            onMouseDown={() => select(a)}
                            onMouseEnter={() => setHighlighted(i)}
                            style={{
                                padding: "9px 14px", cursor: "pointer", fontSize: 13.5,
                                display: "flex", justifyContent: "space-between", alignItems: "center",
                                background: i === highlighted ? "rgba(56,189,248,0.1)" : "transparent",
                                color: "var(--text-primary)",
                                transition: "background 0.1s",
                            }}
                        >
                            <span>{a.name}</span>
                            <span style={{
                                fontSize: 11, fontWeight: 700, color: "var(--sky)",
                                background: "var(--sky-muted)", padding: "2px 7px",
                                borderRadius: 4, letterSpacing: 0.5,
                            }}>{a.code}</span>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}
