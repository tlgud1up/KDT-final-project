package com.example.resource.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MorphingEffect {
    @JsonProperty("origin_img")  // 원본 이미지
    private String originImg;

    @JsonProperty("opencv_pro1")
    private String opencvPro1;
    @JsonProperty("opencv_pro2")
    private String opencvPro2;
    @JsonProperty("opencv_pro3")
    private String opencvPro3;
    @JsonProperty("opencv_pro4")
    private String opencvPro4;
    @JsonProperty("opencv_pro5")
    private String opencvPro5;

}
