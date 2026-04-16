declare module 'react-simple-maps' {
    import { ComponentType, ReactNode, CSSProperties, SVGAttributes } from 'react';

    export interface ComposableMapProps {
        projection?: string;
        projectionConfig?: Record<string, any>;
        width?: number;
        height?: number;
        style?: CSSProperties;
        children?: ReactNode;
    }

    export interface GeographiesProps {
        geography: string | Record<string, any>;
        children: (data: { geographies: any[] }) => ReactNode;
    }

    export interface GeographyProps {
        geography: any;
        fill?: string;
        stroke?: string;
        strokeWidth?: number;
        style?: {
            default?: CSSProperties;
            hover?: CSSProperties;
            pressed?: CSSProperties;
        };
    }

    export interface MarkerProps {
        coordinates: [number, number];
        children?: ReactNode;
        onMouseEnter?: (event: any) => void;
        onMouseLeave?: (event: any) => void;
    }

    export interface LineProps {
        from: [number, number];
        to: [number, number];
        stroke?: string;
        strokeWidth?: number;
        strokeLinecap?: string;
    }

    export const ComposableMap: ComponentType<ComposableMapProps>;
    export const Geographies: ComponentType<GeographiesProps>;
    export const Geography: ComponentType<GeographyProps>;
    export const Marker: ComponentType<MarkerProps>;
    export const Line: ComponentType<LineProps>;
}
