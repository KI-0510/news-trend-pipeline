 const reportTypeSelect = document.getElementById('reportType');
const reportDateSelect = document.getElementById('reportDate');
const reportTimeSelect = document.getElementById('reportTime'); // 시간 Select
const reportFileSelect = document.getElementById('reportFile'); // 파일 Select
const reportFrame = document.getElementById('reportFrame');
let reportIndexData = {};

async function fetchReportIndex() {
    try {
        const response = await fetch('report_index.json'); // 생성된 인덱스 파일 로드
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        reportIndexData = await response.json();
        populateReportTypes(); // 초기 로드 시 타입 채우기
        populateDates(); // 초기 로드 시 날짜 채우기 (기본값: daily)
    } catch (error) {
        console.error('Error fetching report index:', error);
        reportDateSelect.innerHTML = '<option>리포트 목록 로드 실패</option>';
    }
}

function populateReportTypes() {
    // 이미 HTML에 타입이 정의되어 있으므로, 이 함수는 필요시 확장용
    reportTypeSelect.value = 'daily'; // 기본값 설정
}

function populateDates() {
    const selectedType = reportTypeSelect.value;
    reportDateSelect.innerHTML = ''; // 기존 날짜 옵션 초기화
    reportTimeSelect.innerHTML = ''; // 시간 옵션 초기화
    reportFileSelect.innerHTML = ''; // 파일 옵션 초기화
    reportFrame.src = 'about:blank'; // 프레임 초기화

    const dates = reportIndexData[selectedType] ? Object.keys(reportIndexData[selectedType]) : [];

    if (dates.length === 0) {
        reportDateSelect.innerHTML = '<option>선택 가능 날짜 없음</option>';
        return;
    }

    // 날짜는 이미 최신순으로 정렬되어 있다고 가정 (generate_report_index.py 에서 처리)
    dates.forEach(date => {
        const option = document.createElement('option');
        option.value = date;
        option.textContent = date;
        reportDateSelect.appendChild(option);
    });

    if (dates.length > 0) {
        reportDateSelect.value = dates[0]; // 가장 최신 날짜를 기본 선택
        populateTimes(); // 날짜 선택 후 시간 채우기
    }
}

function populateTimes() {
    const selectedType = reportTypeSelect.value;
    const selectedDate = reportDateSelect.value;
    reportTimeSelect.innerHTML = ''; // 기존 시간 옵션 초기화
    reportFileSelect.innerHTML = ''; // 파일 옵션 초기화
    reportFrame.src = 'about:blank'; // 프레임 초기화

    const timeEntries = reportIndexData[selectedType]?.[selectedDate] || [];

    if (timeEntries.length === 0) {
        reportTimeSelect.innerHTML = '<option>선택 가능 시간 없음</option>';
        return;
    }

    // 시간은 이미 최신순으로 정렬되어 있다고 가정
    timeEntries.forEach(entry => {
        const option = document.createElement('option');
        option.value = entry.time;
        option.textContent = entry.time;
        reportTimeSelect.appendChild(option);
    });

    if (timeEntries.length > 0) {
        reportTimeSelect.value = timeEntries[0].time; // 가장 최신 시간 기본 선택
        populateFiles(); // 시간 선택 후 파일 채우기
    }
}


function populateFiles() {
    const selectedType = reportTypeSelect.value;
    const selectedDate = reportDateSelect.value;
    const selectedTime = reportTimeSelect.value;
    reportFileSelect.innerHTML = ''; // 기존 파일 옵션 초기화
    reportFrame.src = 'about:blank'; // 프레임 초기화

    const timeEntries = reportIndexData[selectedType]?.[selectedDate] || [];
    const selectedEntry = timeEntries.find(entry => entry.time === selectedTime);
    const reports = selectedEntry?.reports || [];

    if (reports.length === 0) {
        reportFileSelect.innerHTML = '<option>선택 가능 리포트 없음</option>';
        return;
    }

    let defaultReportPath = '';
    reports.forEach(report => {
        const option = document.createElement('option');
        option.value = report.path; // 값은 전체 경로
        option.textContent = report.name; // 보이는 것은 파일 이름
        reportFileSelect.appendChild(option);
        // 기본 리포트 (예: commentary 아닌 기본 report) 경로 설정
        if (!report.name.includes('commentary') && report.name.endsWith('.html')) {
            defaultReportPath = report.path;
        }
    });

     // 기본 리포트가 있으면 그것을 선택, 없으면 첫 번째 리포트 선택
    reportFileSelect.value = defaultReportPath || reports[0]?.path;
    loadReport(); // 파일 목록 채운 후 리포트 로드
}

function loadReport() {
    const selectedReportPath = reportFileSelect.value;
    if (selectedReportPath) {
        // report_index.json에 저장된 outputs/... 경로를 그대로 사용
        reportFrame.src = selectedReportPath;
    } else {
        reportFrame.src = 'about:blank';
    }
}

// Event listeners
reportTypeSelect.addEventListener('change', populateDates);
reportDateSelect.addEventListener('change', populateTimes);
reportTimeSelect.addEventListener('change', populateFiles); // 시간 변경 시 파일 목록 업데이트
reportFileSelect.addEventListener('change', loadReport); // 파일 변경 시 리포트 로드

// Initial load
fetchReportIndex();
