package com.example.resource.dto;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Setter
@NoArgsConstructor
@AllArgsConstructor
public class MorphingRequest {
    private Integer h;
    private Integer s;
    private Integer v;

    public int getH() { return h != null ? h : 0; }
    public int getS() { return s != null ? s : 0; }
    public int getV() { return v != null ? v : 0; }
}