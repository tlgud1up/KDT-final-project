
let currentChartInstances = {
    donut: null,
    bar: null
};


async function openAnalysisModal(resultId) {
    try {
        //백엔드에서 해당 ID의 분석 결과 데이터 가져오기
        const response = await fetch(`/result/${resultId}`, {
            method: 'GET',
            credentials: 'include'
        });

        if (!response.ok) {
            throw new Error('결과 데이터를 가져올 수 없습니다.');
        }

        const result = await response.json();
        displayAnalysisModal(result);

    } catch (error) {
        console.error('모달 열기 오류:', error);
        alert('분석 결과를 불러올 수 없습니다.');
    }
}

function displayAnalysisModal(result) {
    const modal = document.getElementById('analysisModal');
    if (!modal) {
        console.error('분석 모달을 찾을 수 없습니다.');
        return;
    }

    // 분석일시 표시
    const analysisDateElement = document.getElementById('modalAnalysisDate');
    if (analysisDateElement) {
        analysisDateElement.textContent = result.analysisDate || '-';
    }

    // 적합/부적합 라벨 색상 변경
    const distinctionLabel = document.getElementById('modal_distinction_label');
    if (distinctionLabel) {
        distinctionLabel.textContent = result.suitable || '판정 없음';
        distinctionLabel.style.color = (result.suitable === "적합") ? "#2f9055" : "#b52b09";
    }

    // 적합/부적합 배지 색상 변경
    const distinctMark = document.getElementById('modal_distinction_mark');
    if (distinctMark) {
        distinctMark.textContent = result.suitable || '-';
        distinctMark.style.backgroundColor = (result.suitable === "적합") ? "#2f9055" : "#b52b09";
    }

    // 불순물 비율 및 개수 표시
    const recapNumber = document.getElementById('modal_recap_number');
    const recapRatio = document.getElementById('modal_recap_ratio');

    if (recapNumber) {
        recapNumber.innerHTML = `<strong>${result.count || 0}개</strong>`;
    }

    if (recapRatio) {
        recapRatio.innerHTML = `<strong>${result.total || 0}%</strong>`;
    }

    // 불순물 비율에 따른 색상 변경
    const total = Number(result.total || 0);
    const recapTexts = document.querySelectorAll('.modal_recap_text');
    recapTexts.forEach(p => {
        if (total > 1) {
            p.style.color = "#b52b09";
        } else {
            p.style.color = "#2f9055";
        }
    });

    const recommendationText = document.getElementById('modal_recommendation_text');
    if (recommendationText) {
        if (total > 1) {
            recommendationText.innerHTML = `검출된 불순물 비율(${total}%)이 KS기준(1.0% 이하)을 초과합니다.<br>
            추가 선별 과정을 통해 불순물 제거 후 재검사를 권장합니다.`;
        } else {
            recommendationText.innerHTML = `검출된 불순물 비율(${total}%)이 KS기준(1.0% 이하)을 만족합니다.<br>
            추가 선별 과정없이 진행하셔도 좋습니다.`;
        }
    }

    updateRatioBars(result);

    const modalOrigImage1 = document.getElementById('modalOrigImage1');
    const modalRcnnResult = document.getElementById('modalRcnnResult');

    if (modalOrigImage1) modalOrigImage1.src = result.originalImagePath || '';
    if (modalRcnnResult) modalRcnnResult.src = result.rcnnResultPath || '';

    const modalOrigImage2 = document.getElementById('modalOrigImage2');
    const modalOpencvPro = document.getElementById('modalOpencvPro');
    const modalOpencvResult = document.getElementById('modalOpencvResult');

    if (modalOrigImage2) modalOrigImage2.src = result.originalImagePath || '';
    if (modalOpencvPro) modalOpencvPro.src = result.opencvProPath || '';
    if (modalOpencvResult) modalOpencvResult.src = result.opencvResultPath || '';

    const modalPca = document.getElementById('modalPca');
    if (modalPca) modalPca.src = result.pcaPath || '';

    destroyModalCharts();

    createModalDonutChart(result);

    createModalBarChart(result);

    modal.style.display = 'block';
}

/* 불순물 종류별 비율 바 업데이트*/
function updateRatioBars(result) {
    const totalImpurities = (result.vinyl || 0) + (result.plastic || 0) + (result.wood || 0);

    if (totalImpurities === 0) return;

    const ratioVinyl = (result.vinyl / totalImpurities * 100).toFixed(1);
    const ratioPlastic = (result.plastic / totalImpurities * 100).toFixed(1);
    const ratioWood = (result.wood / totalImpurities * 100).toFixed(1);

    // 비닐 바
    const vinylBar = document.querySelector('.modal_ratio_box.vinyl .modal_ratio_value');
    const vinylPercent = document.querySelector('.modal_ratio_box.vinyl .modal_ratio_percent');
    if (vinylBar) vinylBar.style.width = `${ratioVinyl}%`;
    if (vinylPercent) vinylPercent.textContent = `${ratioVinyl}%`;

    // 플라스틱 바
    const plasticBar = document.querySelector('.modal_ratio_box.plastic .modal_ratio_value');
    const plasticPercent = document.querySelector('.modal_ratio_box.plastic .modal_ratio_percent');
    if (plasticBar) plasticBar.style.width = `${ratioPlastic}%`;
    if (plasticPercent) plasticPercent.textContent = `${ratioPlastic}%`;

    // 목재 바
    const woodBar = document.querySelector('.modal_ratio_box.wood .modal_ratio_value');
    const woodPercent = document.querySelector('.modal_ratio_box.wood .modal_ratio_percent');
    if (woodBar) woodBar.style.width = `${ratioWood}%`;
    if (woodPercent) woodPercent.textContent = `${ratioWood}%`;
}

/* 모달 도넛 차트 생성 */
function createModalDonutChart(result) {
    const donutCanvas = document.getElementById("modal-donut-chart");
    if (!donutCanvas) return;

    const totalImpurities = (result.vinyl || 0) + (result.plastic || 0) + (result.wood || 0);
    const ratioVinyl = totalImpurities > 0 ? (result.vinyl / totalImpurities * 100) : 0;
    const ratioPlastic = totalImpurities > 0 ? (result.plastic / totalImpurities * 100) : 0;
    const ratioWood = totalImpurities > 0 ? (result.wood / totalImpurities * 100) : 0;

    const donutCtx = donutCanvas.getContext('2d');
    currentChartInstances.donut = new Chart(donutCtx, {
        type: 'doughnut',
        data: {
            labels: ["폐비닐", "폐플라스틱", "폐목재"],
            datasets: [{
                data: [ratioVinyl, ratioPlastic, ratioWood],
                backgroundColor: ['#3b4d08', '#303e51', '#be563d']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'right',
                    align: 'start',
                    labels: {
                        boxWidth: 12,
                        boxHeight: 12,
                        usePointStyle: false,
                        pointStyle: 'rect',
                        padding: 10,
                        color: '#000'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed.toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
}

/*모달 바 차트 */
function createModalBarChart(result) {
    const barCanvas = document.getElementById("modal-bar-chart");
    if (!barCanvas) return;

    const barCtx = barCanvas.getContext('2d');
    currentChartInstances.bar = new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: ["폐비닐", "폐플라스틱", "폐목재"],
            datasets: [
                {
                    label: "전체 이미지 불순물 평균 검출량",
                    data: [result.avgVinyl || 0, result.avgPlastic || 0, result.avgWood || 0],
                    backgroundColor: '#f1512e'
                },
                {
                    label: "해당 이미지 불순물 검출량",
                    data: [result.vinyl || 0, result.plastic || 0, result.wood || 0],
                    backgroundColor: 'black'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: {
                        boxWidth: 12,
                        boxHeight: 12,
                        usePointStyle: false,
                        pointStyle: 'rect',
                        padding: 10,
                        color: '#000'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

/*모달 차트 제거 */
function destroyModalCharts() {
    if (currentChartInstances.donut) {
        currentChartInstances.donut.destroy();
        currentChartInstances.donut = null;
    }
    if (currentChartInstances.bar) {
        currentChartInstances.bar.destroy();
        currentChartInstances.bar = null;
    }
}

function closeAnalysisModal() {
    const modal = document.getElementById('analysisModal');
    if (modal) {
        modal.style.display = 'none';
    }
    destroyModalCharts();
}

function initModalEventListeners() {
    // 닫기 버튼
    const closeBtn = document.getElementById('modal-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', closeAnalysisModal);
    }

    // 배경 클릭 시 닫기
    window.addEventListener('click', (e) => {
        const modal = document.getElementById('analysisModal');
        if (e.target === modal) {
            closeAnalysisModal();
        }
    });

    //새 분석 버튼 (있을 경우)
    const newAnalyzeBtn = document.getElementById('modal_new_analyze_button');
    if (newAnalyzeBtn) {
        newAnalyzeBtn.addEventListener('click', function() {
            window.location.href = '/analyze';
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initModalEventListeners();
});