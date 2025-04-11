import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { useState, useMemo } from 'react';
import { View, Dimensions, Text } from 'react-native';
import { Calendar, LocaleConfig } from 'react-native-calendars';

const { width } = Dimensions.get('window');

LocaleConfig.locales['ko'] = {
  monthNames: [
    '1월',
    '2월',
    '3월',
    '4월',
    '5월',
    '6월',
    '7월',
    '8월',
    '9월',
    '10월',
    '11월',
    '12월',
  ],
  monthNamesShort: [
    '1월',
    '2월',
    '3월',
    '4월',
    '5월',
    '6월',
    '7월',
    '8월',
    '9월',
    '10월',
    '11월',
    '12월',
  ],
  dayNames: ['일요일', '월요일', '화요일', '수요일', '목요일', '금요일', '토요일'],
  dayNamesShort: ['일', '월', '화', '수', '목', '금', '토'],
  today: '오늘',
};

LocaleConfig.defaultLocale = 'ko';

export default function NewsletterCalendar() {
  const nowUTC = new Date();
  const nowKST = new Date(nowUTC.getTime() + 9 * 60 * 60 * 1000); // 한국시간
  const todayString = nowKST.toISOString().split('T')[0];
  const isBefore6PM = nowKST.getHours() < 18;

  const maxDateString = isBefore6PM
    ? new Date(nowKST.getTime() - 24 * 60 * 60 * 1000).toISOString().split('T')[0]
    : todayString;
  const minDateString = '2025-04-01';

  const [currentDate, setCurrentDate] = useState(todayString);
  const router = useRouter();

  const isSameMonth = (curDate: string, targetDate: string) => {
    return (
      new Date(curDate).getFullYear() === new Date(targetDate).getFullYear() &&
      new Date(curDate).getMonth() === new Date(targetDate).getMonth()
    );
  };

  const onPressDate = (date: any) => {
    const newDate = date.dateString.replaceAll('-', '').slice(2);
    router.navigate(`${ROUTE.NEWSLETTER.INDEX}/${newDate}`);
  };

  const markedDates = useMemo(() => {
    return {
      [todayString]: {
        customStyles: {
          container: {},
          text: {
            color: '#724EDB',
            fontWeight: 'bold',
            fontSize: 16,
          },
        },
      },
    };
  }, [todayString]);

  const onGoBack = () => {
    router.navigate(ROUTE.HOME);
  };

  return (
    <>
      <CustomHeader title="뉴스레터" onGoBack={onGoBack} />
      <View className="flex-1 items-center justify-center gap-6 pb-20">
        <Text className="text-lg font-bold text-text">
          날짜를 클릭하면 뉴스레터를 볼 수 있어요!
        </Text>
        <Calendar
          onDayPress={onPressDate}
          markingType="custom"
          markedDates={markedDates}
          renderHeader={(date: Date) => {
            const year = date.getFullYear();
            const month = date.getMonth() + 1;
            return (
              <Text
                style={{
                  fontSize: 20,
                  fontWeight: 'bold',
                  color: '#111827',
                  textAlign: 'center',
                  paddingVertical: 10,
                  marginBottom: 8,
                }}>
                {`${year}년 ${month}월`}
              </Text>
            );
          }}
          style={{
            width: width * 0.9,
            borderWidth: 1,
            borderColor: '#E5E7EB',
            borderRadius: 16,
            padding: 16,
            backgroundColor: '#ffffff',
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.1,
            shadowRadius: 6,
            elevation: 4,
          }}
          theme={{
            textDayFontSize: 16,
            textDayFontWeight: '500',
            textMonthFontSize: 20,
            textMonthFontWeight: 'bold',
            textDayHeaderFontSize: 14,
            textDayHeaderFontWeight: '600',
            selectedDayBackgroundColor: '#724EDB',
            selectedDayTextColor: '#fff',
            arrowColor: '#724EDB',
            todayTextColor: '#724EDB',
            monthTextColor: '#111827',
            textSectionTitleColor: '#9CA3AF',
            dayTextColor: '#374151',
            textDisabledColor: '#D1D5DB',
          }}
          current={currentDate}
          minDate={minDateString}
          maxDate={maxDateString}
          hideExtraDays
          onMonthChange={(date: any) => {
            const newDate = `${date.year}-${String(date.month).padStart(2, '0')}-01`;
            setCurrentDate(newDate);
          }}
          disableArrowLeft={isSameMonth(currentDate, minDateString)}
          disableArrowRight={isSameMonth(currentDate, maxDateString)}
        />
      </View>
      <CustomFooter />
    </>
  );
}
